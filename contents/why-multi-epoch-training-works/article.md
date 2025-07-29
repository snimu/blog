# Why multi-epoch training works

*Quick thoughts about multi-epoch training; this is mostly a note to myself.*

Given how often you hear that LLMs just memorize the training dataset, why do they benefit from being trained on the same data multiple times? (Note: I don't think that they just memorize the training data, but I'm still curious about the question.) Here's my answer.

## Why epoch 2 is useful

In a sense, training-batches are like tokens in a sequence during inference: the model can make use of the data it has already been updated on (past batches during training / past tokens in the sequence during inference) to contextualize the current information (the current batch / token), but not future information. Therefore, a lot of training data can only be contextualized by the full training data in epoch 2, and thus by all potentially relevant information. And so training for multiple epochs will help the LLM form a more complete world model.

## Why we want a contextualized world model

To demonstrate the importance of having contextualized information in the model itself, consider the case of LLMs using search engines:

- If they have a lot of contextualized world knowledge, they aren't gullible and can distinguish good content from bad, even based on pretty small details
- If they don't, they will fall for every SEO scam around, and those are almost always at the top of the results (that's their purpose)

This is because in order to judge the truth of something, you need to have a bunch of related, known facts with which you can contrast it. And those facts can only be used to check other facts when they themselves are somehow grounded. Therefore, 1) you need a set of facts to avoid gullibility, and 2) that set of facts must itself be contextualized to form a proper world model that doesn't consist of SEO cloaca.

Of course, sometimes we want the models to be gullible; I've often seen screenshots of chats where the LLM doesn't believe a true news story because it's so absurd (and while some of those screenshots are surely fake, I don't think all of them are). The models just acting along here is in conflict with them being able to push back against the user in other scenarios. And while googling will help in this case, it's the exact example that started this section. If the models don't already have a good world model, they are at risk of just loading more wrong facts into their memory from Google. Multi-epoch training means that they effectively always have their entire dataset in memory (very roughly speaking), which helps them judge search results and pick the best ones.

Generally, it's easier to understand a concept if you have mastered the pre-requisites, instead of stumbling randomly upon it with no prior experience in the field. The latter only really allows for memorization.

## Beyond epoch 2

Even having trained for two epochs, there might still be limitations to the contextualization, for the following reasons:

1. The first few training batches of epoch 1 might already so far in the past that they won't be used to contextualized the first few batches of epoch 2 (and since we shuffle, the first batches of epoch 1 aren't the same as those from epoch 2) because of unlearning of old data during trainings
2. At the start of epoch 1, the model might still be so dumb that it cannot really learn any abstract information that could be used for contextualization of newer data. It might instead just be picking up basic syntax at first

Both problems are solved by just training for more epochs. Of course, they would also be solved by simply training on more non-repeated data, but that's not always available. The point is that training for multiple epochs will help the model understand a given dataset better, and is sometimes the only possible way to get more high-quality tokens. (Problem 2&mdash;the model being too dumb at the start of training&mdash;could maybe be softened by pre-pre-training on common token n-grams, a formal language, or other token sequences that can be auto-generated but are not missed when forgotten, but multi-epoch training is just such a simple solution if you have the compute)

## Random sampling vs. curriculum learning

The above implies something about why randomly sampled batches work so well. They benefit from mostly containing non-related samples within one batch, because a model cannot contextualize one sample in a batch with another sample in the same batch. Random sampling makes it likely that related information is presented in separate batches.

For curriculum learning, this implies that it's better if related concepts and information are spread out over the batches rather than collecting them into a single batch each. It might be even better to keep related texts in different but adjacent batches (or maybe do that in the first batch then spread them out in subsequent batches? Or the other way around? I could come up with an explanation for any of these being best so I don't actually know). This part is quite speculative, I'll have to look up the literature at some point (if I can somehow find the time).

## Small-batch training

I've read in multiple places that an LLM trained on smaller batches will be better than one trained on large batches but the same total number of tokens, though for a higher cost. This aligns nicely with models' need for contextualization, so I believe that the ability to contextualize each sample with all the immediately preceding ones is one reason for the superiority of small batches.

## Conclusion

Samples in a batch are unrelated and batches are only available to the LLM sequentially, which means that multi-epoch training improves performance compared to single-epoch training on the same data, and which benefits randomly shuffled datasets (though it is possible to be better).
