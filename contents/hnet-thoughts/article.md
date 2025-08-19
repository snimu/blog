# Various thoughts on the H-Net

I have just read the H-Net paper (LINK) and it's amazing. So amazing in fact that inspired a lot of thoughts around it. I'll put some of those in dedicated articles, and the scattered here.

This article assumes that you've already read the paper, or at least chapters 1 and 2.

I'll split it into two parts: interesting ways to use the H-Net, and scattered thoughts about the architecture.

## Interesting uses for the H-Net

First, two potential use-cases for the H-Net that I didn't see in the paper (I might have simply overlooked them, though): embedding models, and dynamic chunking as a tool for science.

### H-Net as an embedding model

The hierarchical structure of the H-Net means that byte-embeddings get compressed into more compact representations, which are compressed into even more compact ones, and so on. What you get from the cascade of encoders is a set of vectors which are progressively smaller, and encode progressively more abstract information.

That's a perfect setup for late interaction RAG (LINK):

- Doing late interaction search makes full use of the information that is available
- We can do a sort of hybrid search, where we search for very abstract, semantic similarity with some vectors, and more lexigraphic similarity with others
- And that might be used to improve efficiency: the abstract vectors are much smaller than the ones closer to the bytes, so doing late interaction with them is much cheaper. This can be taken advantage of by first filtering out all chunks that aren't a semantic fit, and then searching with more granularity among the remainder. And of course, this  can be done in multiple stages if the hierarchy of the H-Net is deep enough

### Dynamic chunking as a tool for science

The authors have shown that H-Net is amazing as a DNA model. A model that is great at predicting DNA and performing downstream tasks on it is of course very useful (and such an H-Net could be used for embedding search over DNA chunks, [see above](#h-net-as-an-embedding-model)), but there's more:

If the model has multiple levels of abstraction, it will will dynamically merge base pairs into evermore abstract representations. How it groups the base pairs might be very interesting for science!

It might actually make sense to train multiple H-Nets with different initializations and data order on overlapping sets of DNA data (and whatever other modifications you can think of), then comparing the groupings of base pairs (and of more abstract concepts down the hierarchy) between them. Where these grouping overlap, we might find universally meaningful, context-dependent chunks of base pairs which could turn out to be very useful for the field of biology in their own right.

Of course, the same should be possible with a lot of other domains as well, making this a very broad tool for science.

## Thoughts on the architecture

In this section, I'll speculate about the architecture of the H-Net. These thoughts are purely theoretical, but they will inform what experiments I'll try out once I get to it. Feel free to try them out yourself though! I'd appreciate a citation, but I certainly won't get around to trying all of these out, so go ahead. All of them are independent from each other, and none of them are guaranteed to work.

### Improving how the model dimension is changed

The encoder and decoder of the H-Net act on much longer sequences than the main module, so to make up for the increased compute requirement, they have a lower model dimension than the main module.

To achieve this, the model dimension of course has to be actively changed. It has to be increased when stepping down into the main module, and decreased again when stepping back up into the decoder. The typical way to do this would be to use linear layers in between, but those interrupt the residual layer and therefore destroy gradient flow (which is especially important in multi-level hierarchies, but already unacceptable for one level of hierarchy). To avoid this problem, the authors borrow a technique from SpaceByte (LINK): to increase dimensionality, we concatenate the small vector with a learned vector of the right dimension. To reduce dimensionality, we simply cut off and discard the excess vector-entries.

This process preserves gradient flow and thus works well, but I see a potential improvement to both parts (the up- and the down-projection).

#### Down-projection

In the down-projection, I find it inefficient to just throw away a part of the activation vectors. While yes, the model can make use of the full dimensionality in intermediate layers, and concentrate all that information at the part that isn't discarded, it's still lost model capacity (or rather, data-transport capacity, which is also important).

To make up for that, we'll borrow a trick from modded-nanogpt (LINK): adding the input embeddings to the residual at every layer. They perform `x = x_lambda * x + x0_lambda * x0`, where the two lambdas are learned scalars, `x` is the residual at the input to the current layer, and `x0` are the original token embeddings.

We can do the same with the discarded part of our vector!

Imagine that the model dimension at level `s+1` is 2048, and at level `s` it's 1024. Then in the normal H-Net, half of the output vector of the main module is simply discarded. We would instead take that half of the output, and add it to each part of the de-chunked input to the decoder at level `s` in a learned weighted sum.

What if the dimensions don't add up?

Let's say the dimension at level `s+1` is still 2048, but that at level `s` is 1536. Then, we won't cut off a vector of dimension 1024 fitting perfectly the dimension at level `s`; instead we will cut off a vector of dimension 512, and will miss another 1024 entries! Well, we can make up for that by using the same trick that is used to increase vector dimensionality already: concatenate a learned vector. Then, we won't waste the cut-off part of the vector, and the operation will still work.

And if the dimension at level `s+1` is 2048 but that at level `s` is 512, we can simply cut the cut-off vector of dimension 1536 into three vectors of dimension 512 and add each of them at the input of a different layer of the decoder.

#### Up-projection

It makes a lot of sense to concatenate a learned vector when going from level `s` to level `s+1`. It's simple, preserves the gradient, and can add a useful biads to the activations.

However, I think it could be better to make that added vector data-dependent. This is a great opportunity for adding other modalities in (in a video H-Net, you could add audio information here; but I'll get into that in another article). But for a pure text model, we can still find a use for this cutoff: just run a second encoder a model dimension that is equal to the missing dimensionality, on the same data but with only a single layer, chunk the parts in the same way as the main input, and concatenate the results to the chosen input vectors.

This way, we can get add a second view at the data. Looking at modded-nanogpt again, their value embeddings do exactly that, and it seems to work very well. This makes me fairly confident that this addition is valuable. Of course, it only works if using the chunking determined from the main path of the model on the outputs of this second encoder doesn't somehow muck up the model. In my understanding it doesn't, but I'm not completely certain about that.

This of course leaves the second encoder with sparse gradients which only flow into the selected bytes. With enough training, this should still work, but it would be better to get gradient signal for every single input byte. To do this, we could add the output of the second encoder to the residual from the main encoder to the decoder using the same trick as in the [down-projection section](#down-projection) to align their dimensionalities.

### Value embeddings

Speaking of value embeddings, those could be done in the same way as the [up-projection](#up-projection):

- Use a secondary, tiny encoder on the input sequence
- Use the byte-boundaries determined by the main encoder's chunking module
- Repeat this process until you are at the right level of the hierarchy
- Make use of the resulting additional inputs however you want

I suggest using them like the value-embedding in modded-nanogpt, by performing a weighted sum between the attention-values and these additional vectors (the value embeddings), before applying the attention operation. Other uses are possible too, though.

### Improving the residual

There is a residual connection from the output of the encoder to the input of the decoder.

I will first explain its purpose and how exactly it's designed, then state the proposed improvement and two potential upsides to that improvement, then finally discuss some different ways to design it.

#### What the residual is for

It gives the decoder much more fine-grained information, and provides a gradient signal to every byte position. The main module only uses the encoder's output at some byte positions, which are meant to have all the required abstract information about the input encoded in them. However, this means a loss of fine-grained information for the decoder, and that the gradient will only flow through the selected bytes.

The residual fixes both of these problems:

- In the forward pass, I view it as the main input to the decoder, whereas the main module's output act as (highly influential) abstract guidance
- In the backward pass, it means that a gradient flows into every byte position

This residual has a linear layer applied to it. This is required because the output of the encoder fulfills three jobs:

1. It provides the input to the chunking module
2. Selected outputs are used as the inputs to the main module
3. The outputs are used for the residual

The linear layer is used to disentangle the representationsf or these different tasks. While the authors say that it's used to disentangle the representations of the chunker and the residual, that akes little sense to me. The chunker is already disentangled by `Wq` and `Wk`, one linear layer for each of its components. The point seems to be to disentangle the representation of the input and the output of the main module.

The linear layer is placed at the residual instead of the main module because it interferes with the gradient from its input, and since the main module's gradient is far more important than the residual's, we place the linear layer at the residual.

> Remember, the encoder, main module, and decoder are all pure ResNets. There is a continuous residual stream going through all of them, preserving a high-quality gradient. We want to avoid interrupting that gradient as much as we can.

#### The proposed improvement to the residual

My question is: why not try to preserve the continuous residual stream on all branches of the model? Put another Mamba or transformer layer with its own local residual on the residual from the encoder to the decoder, and you achieve just that. You will also get a more expressive transformation, which is nice for inference.

Why the gradient is improved has hopefully been sufficiently explained, so I'll provide more detail on the inference improvements now.

Here is how inference on a 1-stage H-Net works:

1. Assume you have just produced a full token and start from after a boundary
2. Forward pass through all modules, but only produce a single copy of the main module's output, and a single byte output that you sample from that
3. Run the byte through the encoder and chunker again
4. If it's a token boundary, return to 1.
5. If it's not a token boundary, return to 2. to produce a second copy of the main module's hidden state

For a multi-hierarchy model, this process will be incremented once for the single forward pass of the main module. And for the next token at the outermost level of the hierarchy, it will be incremented again, whether or not there was a token boundary in the main module or not.

In any case, we will only call the main module once for every token that we produce, but each token can require multiple passes through the encoder and decoder. If the compression ratio is high, and/or there are many levels in the hierarchy, then the encoders and decoders need to be called a lot of times before the main module is called once. If the vast majority of the parameters are in the main module, that might constitute a problem, because the encoder and decoder might be too dumb to perform their jobs.

As done in the paper, it is useful to have the encoder and decoder at each subsequent level of the hierarchy have more parameters than those in the previous level. In other words, as the sequence length shrinks, compute per token increases every time. This way, the encoder and decoder are powerful enough to produce sensible outputs for many iterations from the same hidden state of the main module.

Replacing the linear layer on the residual from the encoder to the decoder with a more expressive layer will make it fulfill two jobs at once: (1) allow for different representations at the in- and output of the main module, and (2) add more compute along the high-detail path of the model which can be called many times for every time the low-detail path (the main module) is called.

And that's in addition to the improved gradient flow from having a residual in the residual.

#### Mamba or transformer?

The H-Net paper spends a lot of time on the question of what type of mixing-layer to use: a Mamba 2 layer, or a transformer layer consisting of an attention layer with a sliding window of size 1024 and an MLP?

They determine that Mamba is better, because it's better at compressing the local information into the bytes that are passed to the main module. However, they still use attention in the main module, and even in the first and last layer of the second encoder and decoder of the 2-stage H-Net. Clearly, attention is more expressive, it's just worse for the chunking mechanism they chose.

Therefore, I suggest placing a transformer layer with a sliding window on the residual, not a Mamba layer. We don't use the residual's output to chunk anything, so all we're looking for is maximum expressiveness, which attention + MLP provide.

Of course, we can additionally replace the MLP with a sparse MoE, making the layer even more expressive at the same cost.

### Latent Looping

I have several ideas for how to incorporate latent looping into the H-Net, which I think could have multiple advantages. However, I'm hitting the limits of my knowledge very quickly, so the section is somewhat meandering and too speculative even for my taste. I will keep it in, but not as part of the main article.

<details>
<summary>Expand this if you are still interested in my thoughts on latent looping for the H-Net</summary>

I believe that the main module of the H-Net is an attractive target for latent looping a là [Geiping et al.](...) (LINK) and could provide three potential benefits:

- Reduce the variance of memory and compute requirements during training and inference, which are a huge remaining problem of the method
- Handle difficult sections more gracefully
- Encourage more compression

We will go through each of these points one by one, but first, let me quickly explain what I mean by latent looping. I would treat the encoder and decoder like the Prelude and Coda of [Geiping et al.](...) (LINK), and the main module like the recurrent unit:

TODO: IMAGE (of Geiping architecture)

That requires some singificant changes to the architecture, but they are consistent with the hierarchical setup and the dynamic chunking. Whether or not it actually works better than the original is a different question entirely, but since [Geiping et al.](...) (LINK) seem to perform pretty well, there is at least some chance that it does.

And it does have one advantage that is very specific to the H-Net: being a counterweight to the variance in compute- and memory-requirements of the H-Net.

#### Reducing the variance of compute- and memory-requirements

> I have gone through multiple iterations of the ideas in this section, which I'll keep in the order in which I've had them to show multiple approaches that wouldn't work, and why they wouldn't work.

The fact that the H-Net determines token boundaries dynamically means that two sequences of the same length might require very different amounts of memory and compute if one of them is compressed more strongly than the other. The compression ratio of course increases over the course of training, but it will vary almost randomly from batch to batch. If this variance is small, it's not a big issue, but if it's big, then it is. That's because a large variance in compute and memory needs requires you to plan for the least efficient batch possible (or you'll get an Out Of Memory Error, which is very bad). But that means that for most batches, you will have poor GPU usage.

With latent looping, we can make up for that: just loop the main model more often when compression is high, and less often when it is low. If this is done right, it could almost completely make up for compute- and memory-variance by keeping compute per sequence constant.

The problem with this approach is that keeping compute per sequence constant is contra the entire point of the H-Net (or at least its main selling point): When a sequence is simple and thus highly compressible, we *want to* spend less compute on it than if it's difficult.

So instead, we could use the following approach:

- Decide on a baseline number of loops, for example 5
- Depending on the average compression in the current batch, we vary this number up or down
- If it's varied to a non-integer number, we can approximate it

Like in [Geiping et al.](...) (LINK), it would make sense to slowly increase the baseline number over the course of training while reducing the batch size, but that's highly controllable and thus not a problem.

#### Handling difficult sections more gracefully

...

#### Encouraging more compression

...

#### What should we focus on?

... probably on the first point, because it's the biggest limiting factor

#### Inference with latent looping

... can use yet another variable to decide when to stop looping during inference: the similarity between activations from one interation to the next, as done in [Geiping et al.](...) (LINK).

- Latent looping of model
  - Scaling the number of loops in one of two ways:
    1. if p is close to 1.0, increase the number of loops
        - Increasing compute to this token is already taken care of by the smoothing module, by making the next token also consider this token
        - In other words, the higher the probability of a token boundary, the closer to the capacity limit the model *should be* for that token
          - &rarr; needs more compute for tokens with very certain boundaries than uncertain ones
          - &rarr; doing it this way would also allow the model to make the tokens larger because it's a second mechanism for keeping compute per bit of entropy constant
    2. use the number of loops to make up for the amount of compression in the model
        - the model will have varying memory and compute requirements depending on how much a sequence is compressed, which is bad
        - you can simply adjust the amount of latent looping to make up for compute variance, and the number of steps you propagate back with truncated BPTT to make up for memory requirements if they differ
        - this would additionally allow the model to compress more agressively
  - Both of these ways of scaling compute through latent looping are compatible
  - The encoder and decoder act like the Prelude and Coda in Geiping et al. so the techniques from that paper could simply be used on an H-Net

</details>
