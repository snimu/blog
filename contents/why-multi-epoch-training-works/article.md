# Why multi-epoch training works

*Quick thoughts about multi-epoch training; this is mostly a note to myself.*

Why do LLMs benefit from being trained on the same data multiple times? Here's my (non-mathematical, not-rigorous) intuition for why this makes perfeft sense.

## Why epoch 2 is useful

As a rough analogy, training-batches are like tokens in a sequence during inference: the model can make use of the data it has already seen (past batches during training / past tokens in the sequence during inference) to contextualize the current information (the current batch / token), but not of future information. Therefore, a lot of training data can only be contextualized by the full training data, and thus by all potentially relevant information, in epoch 2. And so training for multiple epochs will help the LLM form a more complete world model.

## Why we want a contextualized world model

To demonstrate the importance of having contextualized information in the model itself, consider the case of LLMs using search engines:

- If they have a lot of contextualized world knowledge, they can distinguish good content from bad, even based on pretty small details
- If they don't, they will fall for every SEO scam around

This is because in order to judge the truth of something, you need to have a bunch of related, known facts against which you can contrast it. And those facts can only be used to check other facts when they themselves are somehow grounded. Therefore, 1) you need a set of facts to avoid gullibility, and 2) that set of facts must itself be contextualized to form a proper world model.

Of course, sometimes we want the models to be gullible; LLMs sometimes don't believe a true news story because it's so absurd and that's bad. In this situation, the models should just be acting along, but that's a behavior in conflict with them being able to push back against the user in other scenarios. And while googling could in principle help distinguish the two cases, it's the exact example that started this section. If the models don't already have a good world model, they are at risk of just loading more wrong facts into their memory. Multi-epoch training means that they effectively (very roughly speaking) always have their entire dataset in memory (in a highly compressed and noisy form), which helps them judge search results and pick the best ones.

Generally, it's easier to understand a concept if you have mastered the pre-requisites, instead of stumbling randomly upon it with no prior experience in the field. The latter only really allows for memorization.

## Beyond epoch 2

Even having trained for two epochs, there might still be limitations to the contextualization, because:

1. The first few training batches of epoch 1 might already so far in the past that they won't be used to contextualize the first few batches of epoch 2 because of unlearning of old data during training (and since we shuffle, the first batches of epoch 1 aren't the same as those from epoch 2, therefore the data in the former may be useful for learning the latter)
2. At the start of epoch 1, the model is still pretty dumb, so it might not learn any abstract information that could be used for contextualization of newer data

Both problems are solved by training for more epochs. Yes, they would also be solved by simply training on more non-repeated data as long as it contains some of the same facts, but that's not always available. The point is that training for multiple epochs will help the model understand a given dataset better, and is sometimes the only possible way to get more high-quality tokens. (Problem 2&mdash;the model being too dumb at the start of training&mdash;could maybe be softened by pre-pre-training on common token n-grams, a formal language, or other token sequences that can be auto-generated but are not missed when forgotten, but multi-epoch training is just such a simple solution if you have the compute)

## On random sampling

The same arguments about multi-epoch training helped me build intuition on why randomly sampled batches work so well.

Randomly sampled batches benefit from mostly containing non-related samples within one batch, because a model cannot contextualize one sample in a batch with another sample in the same batch. Random sampling makes it likely that related information is presented in separate batches and can thus build upon itself.

## Small-batch training

One piece of evidence for the need for contextualization is that an LLM trained on smaller batches will be better than one trained on large batches but the same total number of tokens (though for a higher cost), which might be explained by more samples being available for contextualization instead of being shut off from one another by virtue of being in the same batch.

Admittedly, this is neither strong evidence nor the only explanation for the strength of small batches, but it does align nicely with the other observed facts about LLMs described above.

## Conclusion

Samples in a batch are unrelated and batches are only available to the LLM sequentially, which means that multi-epoch training improves performance compared to single-epoch training on the same data, and which benefits randomly shuffled datasets (though it is possible to be better).

None of the explanations in this article are the *only* explanation for the phenomenon they describe, but they do help me reason about LLM training.
