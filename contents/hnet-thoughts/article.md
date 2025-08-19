# Various thoughts on the H-Net

I have just read the H-Net paper (LINK) and it's amazing. So amazing in fact that inspired a lot of thoughts around it. I'll put some of those in dedicated articles, and the scattered here.

This article assumes that you've already read the paper, or at least chapters 1 and 2.

I'll split it into two parts: interesting ways to use the H-Net, and scattered thoughts about the architecture.

## Interesting uses for the H-Net

First, two potential use-cases for the H-Net that I didn't see in the paper (I might have simply overlooked them, though): embedding models and dynamic chunking as a tool for science.

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

The encoder and decoder of the H-Net act on much longer sequences than the main model, so to make up for the increased compute requirement, they have a lower model dimension than the main model.

To achieve this, the model dimension of course has to be actively changed. It has to be increased when stepping down into the main module, and decreased again when stepping back up into the decoder. The typical way to do this would be to use linear layers in between, but those interrupt the residual layer and therefore destroy gradient flow (which is especially important in multi-level hierarchies, but already unacceptable for one level of hierarchy). To avoid this problem, the authors borrow a technique from SpaceByte (LINK): to increase dimensionality, we concatenate the small vector with a learned vector of the right dimension. To reduce dimensionality, we simply cut off and discard the excess vector-entries.

This process preserves gradient flow and thus works well, but I see a potential improvement to both parts (the up- and the down-projection).

#### Donw-projection

In the down-projection, I find it inefficient to just throw away a part of the activation vectors. While yes, the model can make use of the full dimensionality in intermediate layers, and concentrate all that information at the part that isn't discarded, it's still lost model capacity (or rather, data-transport capacity, which is also important).

To make up for that, we'll borrow a trick from modded-nanogpt (LINK): adding the input embeddings to the residual at every layer. They perform `x = x_lambda * x + x0_lambda * x0`, where the two lambdas are learned scalars, `x` is the residual at the input to the current layer, and `x0` are the original token embeddings.

We can do the same with the discarded part of our vector!

Imagine that the model dimension at level `s+1` is 2048, and at level `s` it's 1024. Then in the normal H-Net, half of the output vector of the main model is simply discarded. We would instead take that half of the output, and add it to each part of the de-chunked input to the decoder at level `s` in a learned weighted sum.

What if the dimensions don't add up?

Let's say the dimension at level `s+1` is still 2048, but that at level `s` is 1536. Then, we won't cut off a vector of dimension 1024 fitting perfectly the dimension at level `s`; instead we will cut off a vector of dimension 512, and will miss another 1024 entries! Well, we can make up for that by using the same trick that is used to increase vector dimensionality already: concatenate a learned vector. Then, we won't waste the cut-off part of the vector, and the operation will still work.

And if the dimension at level `s+1` is 2048 but that at level `s` is 512, we can simply cut the cut-off vector of dimension 1536 into three vectors of dimension 512 and add each of them at the input of a different layer of the decoder.

#### Up-projection

It makes a lot of sense to concatenate a learned vector when going from level `s` to level `s+1`. It's simple, preserves the gradient, and can add a useful biads to the activations.

However, I think it could be better to make that added vector data-dependent. This can be done quite easily: just run a second encoder a model dimension that is equal to the missing dimensionality, on the same data but with only a single layer, chunk the parts in the same way as the main input, and concatenate the results to the chosen input vectors.

This way, we can get add a second view at the data. Looking at modded-nanogpt again, their value embeddings do exactly that, and it seems to work very well. This makes me fairly confident that this addition is valuable. Of course, it only works if using the chunking determined from the main path of the model on the outputs of this second encoder doesn't somehow muck up the model. In my understanding it doesn't, but I'm not completely certain about that.

This of course leaves the second encoder with sparse gradients which only flow into the selected bytes. With enough training, this should still work, but it would be better to get gradient signal for every single input byte. To do this, we could add the output of the second encoder to the residual from the main encoder to the decoder using the same trick as in the [down-projection section](#donw-projection) to align their dimensionalities.

- Linear in residual? Or another Mamba layer / transformer layer
  - actually, trafo more likely to work because Mamba is specifically good for chunking, which we don't have to do with this
  - would preserve a residual stream through all paths in the network -> cleaner gradient
  - would also increase expressiveness of the residual connection across M
- Value embeddings?
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
