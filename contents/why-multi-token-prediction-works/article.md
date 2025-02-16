# Why multi-token prediction works

In single token prediction (STP) the model sees on average half the sequence length in tokens at the input, but only a single token at the output, at each position. But the largest training effect comes from what models are forced to produce at the output! Therefore, increasing the number of targets per step increases the amount of training signal from a given batch and thus improves the model.

If we have a context window size ($T$) of $1024$ at the input, then at token position $1$, the model sees a single token as context, at position $1024$ it sees $1024$ tokens at the input, and on average, it will see $512$ tokens at the input. But it will only ever see $1$ token at the output.

## Targets are more relevant for learning than inputs

This is probably a pretty obvious point to many, but it is central to my argument and so I will go into it a bit.

The gradient with which we find the best possible model weights comes from the difference between the model-prediction and the target. Therefore, the number of gradients we can train the model on depends only on the number of predictions it makes and the targets it makes them against (though the quality of the gradient depends on the inputs, of course; we need the model to change its outputs depending on the inputs, and so need inputs for training).

To illustrate this, let's say that the number of outputs didn't matter, and the number of inputs were much more important. Then, we'd only ever update the model on its prediction at the last token position. If we have $1024$ tokens at the input, then at position $1024$, all tokens that the model sees at position $1023$ will be in its context, too. And all that are seen at position $1022$, and so on. Therefore, if the model could learn to produce every token in that sequence from just seeing them at the input and being forced to produce the correct last output, we wouldn't bother calculating gradients for all the token positions. While I (weakly) believe that there is *some* of that going on, we clearly don't do that, because the output tokens are extremely important.

Again, this is pretty obvious, but I needed to say it, because it demonstrates that increasing the number of targets per input will be very impactful, even if there are still much fewer targets than inputs.

## Targets are repeated; will this be a problem?

You might think that repeating outputs (which we are effectively doing) will have strongly diminishing returns compared to seeing new outputs.

There is likely some truth to that, but consider the following things:

1. It is very cheap to do MTP compared to STP (see the amazing DeepSeek-style MTP from [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3)), therefore even small gains in training signal are worth it
2. The targets in MTP are repeated, but with different context every time, so that there is a difference in the input-output function being learned at different token positions
3. We do the same at the inputs, as explained above, so it cannot be that bad (and multi-epoch training is a thing, though this is a worse argument because there, the outputs are seen very far apart)

## MTP is great for long sequences

I hypothesize that MTP is fantastic for long sequences, precisely because it leads to a larger proportion of tokens being seen at the output rather than the input. Conversely, seeing whether or not models trained on MTP are better with long sequences than ones trained on STP is a great way for testing the entire point of this article.

To illustrate this point, consider a context window size of $1024$ again. This means that our full sequence has $1025$ tokens, $1024$ of which are shown at the input and $1024$ at the output, but shifted by $1$. This means that there is one token at the inputs that is never trained against at the output, and vice versa, but we'll ignore those.

Now, the first token will be seen and thus trained with at positions $1$ to $1014$; it is in the context of $1024$ training updates. On the other hand, the last token will be in context only once, when the $1025^{th}$ token is predicted. In general, token $i$ will be seen $C-i$ times (where, again, $C$ is the context window size). This means that early tokens are drilled into the model far more strongly than late tokens (because the input *does* matter, despite what the point of this article is.) Therefore, models will be stronger at early token positions than at late ones.

MTP partially makes up for this. In a sense, MTP means that we shift the ratio of total input- to total output-tokens seen during training towards the output tokens by a factor of tokens-predicted-at-once. Since the output tokens don't have the same bias toward short sequences as the input tokens do, this should help models be better with long sequences.

*Why do output tokens in MTP not cause the same bias towards short sequences as input tokens do?* There are three answers to this question:

1. The bias towards short sequences in a function of the causal mask. This does not apply to the multiple tokens being predicted, because they all have the same ground-truth tokens as context and thus need no masking between them
2. If we have a context window size of $1024$ and a sequence of length $1025$, then that would be a problem: say we predict $5$ tokens at once, then at position $1021$, we could only predict four tokens ahead ($1022$ $1023$, $1024$, and $1025$). At position $1023$, only three (again, up to position $1025$); and so on. But we can simply draw our inputs and targets from sequences of length $S + n$, where $n$ is the number of tokens predicted at once
3. Even if we didn't have sequences much longer than the context window, the problem I've just described would still only appear rarely, because we never predict remotely as many tokens at once as we have at the input (on average); $n \lt S$

Therefore, MTP might at least weaken the bias of LLMs towards short sequences.

## Evidence via other explantations for why MTP works

I've heard multiple times that the reason why MTP works so well in [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3) is that each prediction depends on the previous one, so that the model is taught to predict coherent sequences / thoughts. This is done by decoding the model's last hidden state with the language head for the first token prediction, then applying another attention layer to the last hidden state and decoding that output with the same language head, and so on.

A good comparison might be [Meta's work on multi-token prediction](https://arxiv.org/pdf/2404.19737). They produce the multiple tokens independently from one another, by decoding the last hidden state of the model with different language heads.

If Meta's MTP method works as well as DeepSeek's, then that is evidence that MTP works better than STP due to the model seeing more target tokens, rather than some other explanation.

> It is not super strong evidence, because even in the Meta paper, the stated explanation for the improved performance is explained by the fact that the model will have to learn to anticipate key tokens multiple steps ahead, and thus be able to adapt its next token prediction to those tokens further ahead. However, I'm not sure how much I agree with that explanation; the tokens before the key token aren't the key token itself, so the only prediction that really matters for benchmarks is the one at the key token. I think it is more likely that using MTP means that the key token was *seen more often* than in STP, which makes the model stronger when predicting the key token. So I do consider Meta's method being close in performance to DeepSeek's to be evidence for my argument.

So how do the two methods compare?

| Method | Model size | Dataset size | MBPP pass@1 gain | HumanEval pass@1 gain |
| --- | --- | --- | --- | --- |
| Meta | 7B | 1T (250B for 4 epochs) | +2.4 points | +0.6 points (eyeballed from plot) |
| Meta | 13B | 1T (250B for 4 epochs) | +4.5 points | +1.7 points |
| DeepSeek | 2.4B active, 15.7B total | 1.33T | +1 point | +6.1 points |
| DeepSeek | 20.9B active, 228.7B total | 540B | +0.6 points | +9.2 points |

As you can see, it is not easy to compare the two; they use different dataset sizes (not to mention the 4 epochs vs 1 epoch, or the precise data mixture which I don't know), Meta uses dense models and DeepSeek Mixture of Experts (MoE), and the parameter counts are different no matter how you slice it.

What we can see is that in MBPP, Meta's reported results are a bit better than DeepSeek's; while on HumanEval, DeepSeek's results are significantly better.

This makes me think that the specific layout of DeepSeek's MTP matters, and the number of target tokens is not the only, or even the largest, factor playing a part in the success of MTP. However, it is likely still significant.

## Citation

```bibtex
@misc{snimu2024mtp,
    title={Why multi-token prediction works},
    author={Sebastian M\"uller},
    year={2025},
    month={jan},
    url={https://github.com/snimu/blog/blob/main/contents/why-multi-token-prediction-works/article.md}
}
```
