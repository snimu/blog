# (Proposal) Multi Token Prediction by Splitting Hidden States

I recently had a neat idea for Multi Token Prediction (MTP): splitting the output hidden states of our model into N chunks and predicting a different number of tokens ahead for each chunk using the same language head.

This has an obvious disadvantage: each chunk is now reduced in dimensionality, so the language head will have less to work with and produce less finely tuned predictions.

However, I also see a potentially big advantage: in the MTP prediction methods that I'm aware of, the same output hidden state is always decoded into multiple tokens, which means that it has to contain specific information about both the next token to be decoded, and at the same time enough abstract semantics to decode another token ahead. Splitting the hidden state and decoding each chunk into a different token prediction explicitly avoids this issue.

I will first discuss advantages and disadvantages of two other methods of MTP, compared to the advantages and disadvantages of my method (which I'll call the Split MTP Method). Then, I will present mitigating measures against the issue of reduced hidden state size for the split MTP. Finally, I will bring it together into a DeepSeek-like MTP schema with some changes that I hope are mostly beneficial (but I don't know for sure).

## Comparing with other MTP methods

Over the course of the forward pass, LLMs move the representation of the data in the residual stream from pure, cleanly separated tokens at the input, to very abstract concepts in the middle layers, to concrete tokens (the next-token predictions) at the output again, see here: [Layer by Layer: Uncovering Hidden Representations in Language Models](http://arxiv.org/abs/2502.02013) or [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588).

The MTP techniques that I'm most aware of actively play against this dynamic. I'll focus on two specific techniques: Meta's [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) and [DeepSeek V3](https://arxiv.org/abs/2412.19437).

My main critique of the other methods relies on the fact that the language head is just an inverse embedding layer (yes, there are slight differences, but that's essentially what it is). That's why the embedding layer and language head sometimes share the same weight in order to save parameters. Therefore, in order to get the best possible probability distribution, the optimal hidden state is a superposition of the embeddings of all the output tokens, with their respective weight (when I say embedding I mean the hidden state that causes a token to be assigned probability 1).

On the other hand, the model needs to perform a lot of abstract computation in order to produce the correct superposition of tokens. This is unlikely to happen in such a clean superposition of specific tokens; and more likely in the superposition of both abstract and concrete concepts, of which there may be more or fewer than there are tokens in the vocabulary.

Therefore, the ideal abstract data representation likely doesn't look the same as the ideal output hidden state for predicting the token probabilities, and the two are in conflict.

To get ahead of criticisms of this chain of logic: yes, there is evidence that the differences aren't extreme. After all, we can 1) simply apply the language head to the residual stream at any layer, or 2) train a linear probe at any layer in the layer. Both produce sensible predictions. However, neither achieves the same quality as the actual language head (at next-token prediction; for other downstream tasks, the abstract representations from the middle of the model are often better). And yes, the output hidden state is, in practice, not always a perfect superposition of token embeddings; but the delta is noise compared to the optimal output hidden state.

> Notation: I will call the current token position `S`, and number the input tokens in the input from `1` to `S`; so the next token is token `S+1`, and so on. The model dimension is `D`, and the number of tokens predicted at once is `K`.

### Better and Faster Large Language Models via Multi-token Prediction

*I'll call this the Meta Method.*

In the Meta Method, the authors use the exact same hidden state to make predictions for multiple future tokens. To distinguish between those future tokens, they employ a different language head for each token further predicted; head 1 predicts token `S+1`, head 2 predicts token `S+2`, and so on:

![Better & Faster Large Language Models via Multi-token Prediction: MTP by multiple language heads](images/mtp-meta.png)

There are some close similarities between the Meta Method and the Split MTP Method. I find it difficult to compare both based on just theoretical considerations, but I will give it my best try.

- An obvious advantage of the Split MTP Method is that it only requires a single language head for all predicted tokens, while the Meta Method requires one per predicted token (the language head is also `K` times smaller, but it it is applied `K` times so there's no difference)
- The flip side of that is that the Meta Method has more capacity to extract information about the future tokens than the Split MTP Method
- As I'll discuss at the end of this article, there is a threshold of parameter count below which the Meta Method *worsens* performance, and above which it improves it. Most likely, this threshold will be increased for the Split MTP Method, because the effective hidden state size that the language head sees is so much smaller, and it might require a certain minimum model width to soften the effect
- An advantage of the Meta Method is that each language head can make use of the full hidden state, as opposed to the Split MTP Method; though they also all have to share the same hidden state, which is not the case in the Split MTP Method
- Training dynamics are an open question to me:
  - The Meta Method requires one language head per token prediction from the same hidden state, which might lead to wasteful competition for resources between the different heads
  - In the Split MTP Method, the different chunks all need to share a language head, which could also lead to representational competition

However, I find the latter method more attractive, because it addresses the central critique I've raised above. In the Split MTP Method, the language head is purely for decoding the hidden state. The transformation of the current (input) token into future (output) tokens happens inside the transformer backbone. In the Meta Method, it must happen in the language heads themselves, at least from some perspective. Yes, it will likely happen in part by the backbone making all those predictions and the language heads only extracting the right superposition of features from the hidden state, but there is also stronger pressure to move the prediction job into the language head itself, which leads to a contradiction with the normal way that representations behave inside a transformer.

### DeepSeek's MTP

DeepSeek's Multi Token Prediction makes a prediction of token `S+1` from the transformer hidden states. For token `S+2`, it then concatenates the actual next token at position `S+1` to the output hidden state from which the token, does a linear projection, applies a single transformer block, and decodes the resulting hidden state with the same language head as the previous token to produce a prediction of token `S+2`. Then, that hidden state is again concatenated with the input token again, projected, and so on. See here:

![DeepSeek MTP](images/mtp-deepseek.png)

This means that the DeepSeek MTP method effectively *only* does next-token prediction, but multiple times per input token; and only the prediction of token `S+1` uses the full model, while the subsequent predictions use very cheap models (norm + FC + a single transformer block), but depend on the model's own previous predictions.

This method is very strong and, let's be honest, is probably superior to what I'm proposing here. However, I still see an issue: The first predicted token is decoded directly by the language head *from the same hidden state that gets used for the further token predictions*. This forces one of two things to happen:

1. The hidden state must contain information beyond what is relevant for the next token prediction (which is bad because that abstract information should in the middle of the transformer, or the language head will have more difficulty decoding the hidden state)
2. The prediction of token `S+2` is based only on the prediction of `S+1` and the true `S+1`

The second option is likely very good for training dynamics, because having to make a good second-token-ahead prediction from the actual next token and the model's own prediction of the next-token-ahead forces the model to make the representation of token `S+1` damn good. I assume that in the extreme case where the hidden state is literally only an optimized superposition of tokens for prediction of token `S+1`, this second prediction would give another signal to make the probability distribution very strong (though otherwise, it encourages some additional, abstract information to be in the hidden state, which conflicts with the goal of predicting token `S+1`).

I definitely want a piece of that for my method, and will write about how to get it [further down](#catching-up-to-deepseek); my problem with the method is that the main transformer backend adds so little to all the token predictions except for the first one, because of the conflict between the first hidden state being useful for predicting both token `S+1` and `S+2`.

In other words, DeepSeek's MTP prediction works because it gives a very strong signal to optimize the immediate next-token prediction so that it makes downstream text easier to predict. It would be nice if it also encouraged a strong abstract representation in the middle of the model more directly. To be clear, it probably does, because the hidden state probably contains a bunch of abstract information to help the subsequent token predictions, but that has its own issues.

### Advantages of the Split MTP Method

The split MTP method explicitly assigns different jobs to different parts of the output hidden state: predicting a different number of tokens ahead. This way, it avoids the issue of the hidden state having to contain both explicit information about the next token and enough abstract information to allow extracting tokens further down the line.

Another advantage of split MTP is that MTP is very simple and requires no extra parameters (at least in this simple configuration). The Meta MTP Method requires extra language heads, and DeepSeek's method extra transformer blocks. This isn't a big deal, but I'll take it.

What's also nice is that our shared language head is smaller, which saves us a lot of parameters. However, it's of course also the cause of the one big disadvantage of this method: reduced fidelity of the language head.

## Mitigating the reduced hidden state size

I can think of two options for combatting the issue of reduced hidden-state size: ["Project and Split"](#project-and-split) and ["Uneven Split"](#uneven-split).

### Project and Split

Typically, the output hidden state, a.k.a. the residual stream after the last transformer block, is directly projected into token space by the language head. If we want a bigger vector to project from, we can simply put another Fully Connected (FC) layer between the two.

At first, it is unclear to me if this would help at all. On the one hand, two linear transformations in a row always have an equivalent single linear transformation (if start and end size of the vector being transformed stays the same), so we should expect no mathematical advantage. On the other hand, it makes the transformation more gradual which SGD-variants like, and it adds more parameters (compared to splitting without projecting first) which also always make learning easier.

Additionally, we can simply put an activation function between the two FC layers, like ReLU^2. This would make the whole construct way more expressive by turning it into a proper 2-layer MLP.

Now we have the issue that these additional parameters are costly compared to a normal transformer. However, let's say we predict 4 tokens ahead, and thus split the hidden state into 4 equal chunks. Then, as long as we don't project the hidden state by less than a factor of 4, we will actually reduce the size of the language head, which is very relevant in terms of parameter count. That's because the language head is of size `output_dim * vocab_size`, and the `vocab_size` is typically huge. So every little increase in size of the output vector adds `vocab_size` many parameters. Our new FC layer, on the other hand, is pretty small: it projects from `hidden_state_dim / 4` into `output_dim`, which are both significantly smaller than any language head dimension in a regular transformer. In total, this mean that we might even still save a few parameters while allowing for MTP from a hidden state that's almost as big as in a normal transformer.

One potential problem remains: how does this affect the gradient? The transformer is intentionally designed to only ever add to the residual stream to make the gradient as crisp as possible, except for the language head at the end. Now we add a second layer at the end. Especially in a low-precision regime, that might harm gradient quality. However, I don't know if it does, and it would be interesting to measure.

### Uneven Split

The most important token to predict is the next token, not the second, third, or fourth token ahead. So what if we simply split the hidden state unevenly? For example, if `D=1024` and `K=4`, we could split the hidden state into chunks of sizes `[512, 256, 128, 128]` for tokens `S+1` to `S+4` ahead. Then, we've only halved the chunk size for token `S+1`, instead of quartering it. This is still not ideal&mdash;and it comes with a disadvantage for the prediction of tokens `S+3` and `S+4`&mdash;but it's still better than before.

But hold up, now the input size to the language head differs between the different token predictions. So do we have to use more than one language head now? Not necessarily: we might get away with simply using a Matryoshka language head, meaning that we only use the first `chunk.size(-1)` features of the language head to make a prediction for that chunk, and thus at least partially share the language head.

This isn't quite ideal; I vaguely remember hearing that Matryoshka-*anything* is always a tradeoff between sharing functionality and the quality of the thing itself; e.g. for Matryoshka embeddings, they allow you to get usable embeddings of many different sizes out of the same vector, but I believe that the full vector embeddings aren't quite as high quality as the equivalent non-Matryoshka full vector. I'm not sure about this, though, and I think that this method might be worth a try.

Of course, this method can be combined with the [Project and Split](#project-and-split) method. First project, then split unevenly; this way, we could get a hidden state for the next-token prediction that is larger than the `model_dim`, and still usable hidden state sizes for the other predictions, or whatever setup we want.

### Catching Up to DeepSeek

The great advantage of DeepSeek's MTP method is, as far as I can tell, that prediction of `S+2` relies on the model's own prediction of `S+1` and can thus be backpropagated through both, which provides an additional training signal per token *for the next-token prediction*, making the next-token prediction stronger. The disadvantage is that the output hidden state of the next-token prediction is used, which either means less learning or worse decoding or both.

We can get (some of) that advantage without the disadvantage by adapting the split MTP method:

- Split the output hidden state into four chunks, `c1`-`c4`
- Predict first token from first chunk, which is also the first hidden state `h1` (`c1==h1`)
- Then concatenate first hidden state `h1` and second chunk `c2`
- Normalize, FC, transformer layer -> second hidden state `h2`
- Predict second token from `h2`
- Repeat with `h2` and third chunk `c3`, etc.

This provides us with two information paths for the second token to predict: `h1` and `c2`. That's important because it allows `h1`&mdash;from which the next token is predicted&mdash;to only contain the information needed to predict the next token, a.k.a. a superposition of token-embeddings, while using `c2` to provide rich, abstract information for the subsequent prediction. Because there is a linear projection followed by a transformer layer between `[h1, c2]` and the prediction of `S+2`, the model has plenty of time to transform that abstract representation into a concrete superposition of embeddings, so this split is actually possible (and likely encouraged).

This means that:

- The conflict between abstract and specific hidden state representation is reduced
- The model can provide way more information to the prediction of `S+2` than predicted probability distribution for token `S+1` and the actual token `S+1`

To add to all that, we retain DeepSeek's advantage of prediction `S+2` to be backpropagated through prediction `S+1`, making the latter stronger through an additional but meaningfully different training signal.

Of course, our great disadvantage remains: the hidden-state size per token is reduced significantly, and while we can of course combine this with the techniques presented above ([Project and Split](#project-and-split) and [Uneven Split](#uneven-split)), the high quality small hidden state size might still be worse than the full hidden state with a conflict between decoding and providing information to downstream predictions.

Additionally, DeepSeek's method for MTP works with teacher forcing at every predicted token, while my method doesn't. Whether that's good or bad is unclear to me. On the one hand, having the true token `S+1` available for predicting token `S+2` prevents an error cascade from prediction `S+1` to prediction `S+2`. On the other hand, predicting token `S+2` only from the hidden states that were produced from tokens `1` to `S` is closer to autoregressive prediction, which might be beneficial. And if my method doesn't work as presented above, it's possible to just append the embedding of the true token S+1 to h1 and c1 (or even train that way with a WSD schedule to get the model's error rate below some threshold, decay the learning rate, then remove the concatenated token S+1 and warm up again and train so that the model is brought closer to the downstream task of autoregressive prediction).

## Summary and Outlook

I have introduced a method for Multi Token Prediction (MTP) by splitting the output hidden state of a language model and using the same language head to predict a different future token from each of the chunks.

This method has the great disadvantage of reducing the hidden state size for predicting the next tokens. While I've introduced two possible mitigating measures in 1) first projecting up, applying a non-linearity, and then splitting, and 2) splitting unevenly if some token predictions are more important to us than others, it is unclear if they actually work and if they do, how well.

I have then adapted the split MTP method I've introduced to gain the advantages of DeepSeek's MTP method without the disadvantages (which are pure speculation from my side, to be clear).

As an outlook, here is the plot of Meta's results for different model sizes (all trained for at least 200B bytes, so around 30-40B tokens):

![Meta Method: results for different model sizes](images/meta-mtp-results.png)

Clearly, (their method of) MTP makes model performance worse unless the model is pretty large. DeepSeek does their ablations on MTP with even larger models, and on more data. What that tells me is that I don't have the money to properly try my method out, so I won't. I'd be very happy though if somebody else tried their hands on it!

## Citation

```bibtex
@misc{snimu2025mtp,
    title={(Proposal) Multi Token Prediction by splitting hidden states},
    author={Sebastian Nicolas Müller},
    year={2025},
    month={July},
    url={https://snimu.github.io/2025/07/18/split-mtp.html}
}
```
