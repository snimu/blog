# Tokenization and batch-norm: incorporating global statistics

*Status: un-researched speculation*

## TLDR

Both batch-norm and tokenization externalize the task of computing global statistics on the dataset, and provide them to the corresponding model. This saves model capacity and introduces a helpful inductive bias.

## Batch-norm

Batch-norm takes the norm over the activations at every layer, for each batch. It uses the mean and std of the entire batch, not of each individual sample, to normalize the activations.

Per-sample norms like layer-norm, on the other hand, normalize values independently for each sample in the batch, so that each sample has a mean and std of $1$ and $0$. This lets the weights focus on the relative sizes of the entries of an input tensor, not their absolute values. These relative sizes carry almost all the information about the input, but can easily be overpowered and hidden by a strong offset or multiplier. Because weights change during training, and can have very high or low absolute values, they can introduce such offsets and multipliers to the activations, which the weights in the next layer will be unable to filter out. Norming does so explicitely, and thus allows much more freedom in the optimization of the weights (because their absolute values are now de-coupled from each other).

So why does batch-norm work better (in CNNs) than per-sample norms?

**By normalizing over the entire batch, each sample in the batch is given information about the global statistics of the batch.** If the mean and std of a sample are close to $1$ and $0$ after batch-normalization, then they were close to those of the average sample of the dataset (assuming a batch is a representative sample for the whole dataset). If they are very different from $1$ and $0$, then the sample is an outlier. And importantly, samples of the same class will be outliers in very similar ways. 

A rough intuition for this is that providing an offset (in the mean and std) from the average sample provides a prior probability to the model, which otherwise would only use the feature-conditional probability. It therefore brings inference closer to Bayesian prediction. While the truth of this statement depends on definitions, I think it provides a reasonable hint at the type of inductive bias that batch-norm introduces.

Of course, if the offset from the average sample is very large, we would remove the advantages of per-sample norms. However, the problematic offsets are ones caused by the absolute values of the weights themselves, and because all inputs to a batch-norm will be computed by the same weights, these offsets will be shared between samples and will be fully removed from every sample, so that only the offsets inherent to the dataset will remain.

I've once thought about computing the dataset-wide mean and std, and passing them to the model along with the input. I would then use the weigths of the model itself to compute the mean and std at each subsequent layer from those before it. By using the weights of the model themselves to compute these statistics, they would automatically adapt as the model is trained. This would be much more efficient than batch-normalization as it is actually done, because it could be done in parallel for each sample in the batch. However, it is unfortunately infeasible (as far as I know), because you cannot compute how the mean and std of a value change with a convolution (and many other operations) from only the mean at the previous layer.

However, this is (almost) exactly what tokenization does!

## Tokenization

When training the tokenizer, we merge symbols based on how often they occur next to each other in the training data. Thus, the tokenizer incorporates the global statistics of the dataset into the data itself, so that we don't have to perform any data sharing between samples in the batch during training or inference.

More precisely, the tokenizer does the step of pattern-matching fine-grained features to more abstract, larger-scale ones. Importantly, it does so deterministically. Learning the same function would take significant capacity if done by the model itself, and it is unclear how well it would learn these statistics. And there are clearly correct tokens to learn. In fact, tokens came first: humans spoke long before they wrote, so tokens are the natural expression of language. T-O-K-E-N is a way to flexibly record the word "token", but the meaning is mapped one-to-one to the word "token", while "knote" is meaningless despite sharing all letters with "token". Tokenizers make this explicit. (The same cannot be said for images, I think.)

### Tokenization issues

You might be surprised to hear me say that tokenization is not only good for efficiency, but also for understanding.

We hear of problems caused by tokenization all the time! And that's true: tokenization introduces biases that aren't always appropriate. For example, in math, the type of global statistics that tokenization carries (how often does this symbol appear right before that one?) aren't very meaningful (because in order to generalize in math, you need to be able to work with every possible combination of digits equally and treat digits independently, so biasing yourself towards certain combinations is actively harmful). However, for other subjects (like general language understanding), I believe that tokenization is very useful, not only because it reduces the sequence length.

> Sidenote: why do byte-level models seemingly work well? I don't know for sure, but they certainly use much longer sequences for the same input than models using tokenizers do. This of course provides a lot of dynamic compute. I wonder, how well would a token-level model perform when using Chain of Thoughts to make its sequence length similar to the byte-level model?

## Citation

If you use this article in your work, please cite:

```bibtex
@misc{muller2024batchnormandtokenization,
    title={Tokenization and batch-norm: incorporating global statistics},
    author={Sebastian M\"uller},
    year={2024},
    month={nov},
    url={https://github.com/snimu/blog/blob/main/contents/tokenization-and-batchnorm/tokenization-and-batchnorm.md}
}
```
