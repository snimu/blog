# The H-Net performs improved Multi Token Prediction

> You should probably read the H-Net paper before reading this article; I won't repeat its content in depth

Understanding the smooting module in the [H-Net](http://arxiv.org/abs/2507.07955) made me view Multi Token Prediction (MTP) from a new perspective: that of keeping the information density of the gradient at each position constant. The H-Net does so explicitly through the smoothing module, while classic MTP does it by averaging the gradients from easy and difficult predictions. In this article, I will present my intuitions on both (and they are just intuitions, not mathematically strict).

## H-Net

After encoding a byte sequence, H-Net assigns each byte a probability score between 0.0 and 1.0. If that score is below 0.5, the byte is not selected, if it is above 0.5, it is selected and treated as a token that represents itself and all the preceeding bytes that weren't selected. The main module (for example, a typical transformer) is run on the selected tokens, and its activations are then de-chunked into the original sequence length to be decoded.

However, right before de-chunking, an additional operation is applied: the smoothing module.

It calculates an exponential moving average (EMA) operation, where the main module output at position `t`, `z[t]`, and the EMA of the output of the main module at position `t-1`, `z_ema[t-1]`, are mixed according to the probability `p[t]` that was assigned to the token boundary of token `t`. This mixing is simply: `z_ema[t] = p[t] * z[t] + (1 - p[t]) * z_ema[t-1]`.

### H-Net: Forward pass

To understand what that does, let's first look at the simple case of two tokens and consider the forward pass:

- If `p[t]` is close to 1.0, then the model was very confident that the previous token ended and the new token begins; and `z_ema[t] = 1 * z[t] + (1 - 1) * z_ema[t-1] = z[t]`, meaning that the previous token has no influence on the current one
- If `p[t]` is close to 0.5, on the other hand, then the model was unsure about the token boundary; and `z_ema[t] ~= (z[t] + z_ema[t-1]) / 2`, meaning that the previous token has as much influence on the prediction at the current position as the current token

If we assume that the model attempts to keep all tokens so long that each prediction has the same difficulty, then a high confidence token boundary means that the next-token prediction from the last token was as difficult as the model could handle, but a low confidence means that the model could handle more prediction in the single forward pass than it was required to do.

The smoothing module makes up for that: since the main module's output at position `t-1` influences the prediction from position `t`, the main module effectively performs multi-token prediction at position `t-1`. That in turn allows it to put all of its power into that prediction, even if the prediction was easy. And that includes the power that it doesn't require for the prediction itself, because the output latent will be re-used and can thus contain additional, over-lapping information.

The real trick is that the smoothing module will perform its smoothing smoothly; the more confident the model is in the token boundary, the less capacity it has left over to help with the next-next-token prediction, and the less is the current latent's influence on far-out tokens. There is no hard boundary.

This means that the model will use constant predictive capacity for each token, and can thus always be used to its fullest extend. That's not the case with pre-tokenized transformers! Most tokens are fairly easy to predict from, so the model is limited by its ability to predict the most difficult tokens. Conversly, if a model is capable of predicting the next token from a difficult position, its intelligence will be under-utilized for all easier positions, which will be most.

### H-Net: backward pass

The same will be true for the backward pass:

- If it was difficult to predict the next token at position `t-1`, the gradient produced at position `t` will not be put into the computations at position `t-1` through the smoothing module
- But if it was easy to predict the next token from position `t-1`, a part of the produced gradient at position `t` flows back into the computations at position `t-1`

This is very useful: if a token is easy to predict, then the weights are already well suited to the given token at position `t-1`, and they could use more gradient signal to be updated. But if it was already hard to predict token `t` from token `t-1`, then additional gradient from position `t` will only interfere with the gradient from position `t-1`.

The smoothing module handles both cases perfectly, and to exactly the required degree.

### H-net: Caveat

The first and most important caveat is that I don't have any experimental evidence for these effects, and so don't know if they actually occur.

Secondly, the above is only true in the idealized case in which the model learns to chunk by the difficulty of prediction. However, I believe that this is likely for two reasons:

1. It is inherently useful, so an H-Net that learns to do it will be better than one that doesn't
2. I would guess that the gradients as discussed above will encourage the chunker to learn exactly this pattern (I don't have a super precise model for why though, just a strong gut feeling, so keep that in mind)

A third concern is that I've also simplified the thought experiment to two tokens, but the reality is more complicated with more than one uncertain token. However, in this scenario, the smoothing module is even better because the same effects can span multiple tokens for both forward and backward.

Another caveat is that what I've described is certainly not the only effect; another view is that the smoothing makes up for a fundamental flaw in causal models: token boundaries cannot always be known without knowing the next byte, because there is inherent ambiguity between multiple continuations. However, I don't think that this devalues the effects discussed above.

## Multi Token Prediction

Classic multi-token prediction (MTP) like in [DeepSeek's V3](https://arxiv.org/abs/2412.19437) is extremely useful to LLMs, and helps them learn to predict the immediate next token better. Could the effects discussed above explain this?

### MTP: Forward pass

In the forward pass, predicting the token at position `t-1` doesn't directly help with predicting token `t`, because the latents aren't mixed (this ignores that you can predict multiple tokens at once for efficiency, but for maximum intelligence I haven't heard this being used except for self-speculative decoding).

### MTP: Backward pass

In the backward pass, MTP approximates the effects of the H-Net's smoothing module. The difficulty of predicting the next token varies from position to position. MTP accumulates the gradients for more than one token into each position. Assuming a uniformly random prediction difficulty, and thus a uniformly random information density in the gradient, the accumulation of gradients from multiple positions into one will lead to a more Gaussian distribution of information density in the accumulated gradient, thanks to the central limit theorem. More Gaussian information density means less variable information density compared to uniformly random, because more weight is at the mean position.

And thus, MTP allows the model to learn closer to its maximum per-step learning capacity at every position in the input sequence.

### MTP: Caveats

The automatic, built-in MTP during the forward pass of the main model doesn't apply to normal, token-based MTP schemes.

Additionally, information density of the gradients isn't kept constant by an explicit mechanism, but through sheer force of statistics and averaging, and I expect that to work much worse.
