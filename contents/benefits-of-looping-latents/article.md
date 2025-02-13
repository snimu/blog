# The benefits of looping latents

It has become fashionable to loop latents in transformers (see [here](https://arxiv.org/abs/2412.06769) or [here](https://arxiv.org/abs/2502.05171)). The reason that is typically given is that normal reasoners (DeepSeek R1, OpenAI o1, etc.) decode the output latents and sample a token at every step. They then feed that sampled token back into the model and repeat. Many are saying that a better approach is to directly feed the output latents into the model input at the next step, so that less information is lost at the output.

It has also become fashinable to question the need for this (see [here](https://x.com/stochasticchasm/status/1889912450994352314?s=46) and [here](https://x.com/thenormanmu/status/1884491195545706809?s=46)). The contention is usually that the kv-cache grows quadratically with the number of tokens at the input, and thus can carry over way more information from step to step than the latents ever could. Thus, the argument goes, losing a little information at each step is a negligible cost in return for the improved interpretability of token predictions.

However, I believe that the amount of compute that is wasted by sampling the next token at every step might be significant, depending on the entropy of the model's output distribution.

## Wasted compute

Information in a transformer only ever flows forward through the layers.

Therefore, when going from step S to step S+1, the only way for layer 1 to take advantage of computation done at the last layer (layer L) is through whatever is fed into its input from the previous step. When this isn't an external input (like a user request), it is usually a token sampled from the latent of layer L at step S.

Now, let's consider the kv-cache.

> I will assume that the kv-cache will store all intermediate computation it has access to at every single step. This assumption is a conservative one relative to my argument: if the kv-cache *doesn't* store all intermediate computation, then the amunt of information lost at each step will be even more relevant.

At layer 1, the kv-cache will only every include information about the embeddings of the input tokens. But at layer 2, the residual stream will already contain information about the computation done at layer 1, so the kv-cache at layer 2 can store information from the embeddings, and from the computation done in layer 1; and so on.

Additionally, the amount of computation added at each layer depends on the number of tokens at the input.

> For MLPs, the computation per token is constant with sequence length, so the total compute from MLPs saved in the kv-cache is linear in sequence length; for attention, the amount of compute per token grows linearly with the sequence length, and so the total amount of compute from attention saved in the kv-cache grows quadratically with sequence length; for simplicity I will assume linear compute growth with sequence length, which is approximately true for short sequences. It is also a lower bound, and so any effect I will be discussing further down is likely to be even stronger in reality.

Thus, at layer i, the kv-cache can store the results of i*S layer applications to the input.

So what does that mean for sampling tokens versus looping latents?

## Information loss and entropy

For a very simplified first "intuition pump", assume for a moment that every layer adds as much information as the embedding layer&mdash;every layer adds one token worth of computation. In that case, sampling a token and feeding it back into the model would lose us the equivalent of D tokens worth of computation, where D is the model depth (number of layers).

In reality, this is of course not the case. There is a better way to think about the amount of information that is lost at each step: it is correlated with the entropy of the output distribution.

Let's assume that the language head adds no information to the model's output; in other words, the entire probability distribution over the vocabulary is already contained in the last layer's latent, and the language head simply changes its form. Then, the output latent and the output distribution are equivalent, and when we think about feeding the output latent back into the model, we can similarly think about the output distribution.

> This should be close to the truth, by the way. Assume a model in which the weights of the embedding layer and the language head are tied, which is a common thing to do for saving parameters. Then, applying the embedding layer to the tokens, and the language head right after, will lead right back to the input tokens. Thus, any information gain to the output distribution is contributed by the transformer layers, not the language head. Of course, it is only the case at temperature 1.0; otherwise, you are changing the output distribution in a way that is disconnected from the output latent.

In that case, we can look at two extremes: zero and maximum entropy.

If we have zero entropy in the output distribution, the next token is absolutely certain. Sampling and feeding it back into the language model, that token represents the full distribution and no information is lost.

If, on the other hand, we have maximum entropy, every token in the output distribution is equally likely. If we randomly sample one, we thus loose all information about all ~100k other possible tokens (to be clear, we would have already screwed up big time if it came to that, but the example is illustrative).

This thought experiment makes obvious that the amount of information lost at each step is directly proportional to the entropy of the output distribution: a higher entropy means that more information is lost.

> This of course implies that looping latents when entropy is high, and sampling tokens in regular CoT is best, as I already discussed in my [COCONUT: parallel pre-training](https://github.com/snimu/blog/blob/main/contents/COCONUT-parallel-pretraining/article.md) article, where I propose using an entropy threshold to switch between looping latents and sampling. The only difficulty I see there is that different domains may require different entropy thresholds.

To give a concrete example, let's assume that the highest probability token has a probability of 0.3; if it is being samples, you are throwing away 70% of the probability mass for the next step. If you sample with a temperature greater than 0.0, you might land on an even less likely next token, losing even more information.

## Citation

```bibtex
@misc{snimu2024loopinglatents,
    title={The benefits of looping latents},
    author={Sebastian M\"uller},
    year={2025},
    month={feb},
    url={https://github.com/snimu/blog/blob/main/contents/benefits-of-looping-latents/article.md}
}
