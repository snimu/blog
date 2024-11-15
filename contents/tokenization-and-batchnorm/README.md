# Tokenization and batch-norm: incorporating global statistics

*Status: un-researched speculation*

## TLDR

Both batch-norm and tokenization externalize the task of computing global statistics on the dataset, and provide them to the corresponding model. This saves model capacity and introduces a helpful inductive bias.

## Batch-norm

Batch-norm takes the norm over the activations at every layer, for each batch. It substracts the mean of the entire batch from each element and divides it by the standard deviation (std) of the entire batch.

Why does this work better (in CNNs) than normalizing each sample independently?

By normalizing over the entire batch, each sample in the batch is given information about the global statistics of the batch. If the mean and std of a sample are close to $1$ and $0$ after batch-normalization, then they were close to the average sample of the dataset (assuming a batch is a representative sample for the whole dataset). If they are very different from $1$ and $0$, then the sample is an outlier. And importantly, samples of the same class will be outliers in very similar ways.

In other words, we waste no model capacity on learning the global statistics of the dataset. Instead, we directly inject them at each layer. (Of course, it is also good that we do this at every layer, so that the batch-statistics at that layer adapt correctly as the model is trained.) We only have to store the batch-statistics at each layer for use after training. (This is because at inference, the batch-size might be $1$ or close to it. Then, the batch-norm would always normalize the mean and std to $1$ and $0$, giving the model false information.)

I've long thought about computing the dataset-wide mean and std once, and passing them to the model along with the input. I would then use the weigths of the model itself to compute the mean and std at each subsequent layer from those before it. By using the weights of the model themselves to compute these statistics, they would automatically adapt as the model is trained. This would be much more efficient than batch-normalization as it is actually done, because it could be done in parallel for each sample in the batch. However, it is unfortunately infeasible (as far as I know), because you cannot compute how the mean of a value changes with a convolution (and many other operations) from only the mean at the previous layer (same for std).

However, this is (almost) exactly what tokenization does!

## Tokenization

When training the tokenizer, we merge symbols based on how often they occur next to each other in the training data. Thus, the tokenizer incorporates the global statistics of the dataset into the data itself.

We do not have to norm the samples with that data, because the samples themselves already contain it by construction!

### Tokenization issues

This might sound surprising at first (and I'm not certain that it's true); we hear of problems caused by tokenization all the time! And that's true: tokenization introduces biases that aren't always appropriate. For example, in math, the type of global statistics that tokenization carries (how often does this symbol appear right before that one?) aren't very meaningful (because in order to generalize in math, you need to be able to work with every possible combination of digits anyways, so biasing yourself towards certain combinations is actively harmful). However, for other subjects (like general language understanding), I believe that tokenization is very useful, not only because it reduces the sequence length.

So why do byte-level models work well? I don't know for sure, but they certainly use much longer sequences for the same input than models using tokenizers do. This of course provides a lot of dynamic compute. I wonder, how well would a token-level model perform when using Chain of Thoughts to make its sequence length similar to the byte-level model? Probably pretty well.

## Citation

If you use this article in your work, please cite:

```bibtex
@misc{muller2024batchnormandtokenization,
    title={Tokenization and batch-norm: incorporating global statistics},
    author={Sebastian M\"uller},
    year={2024},
    month={nov},
    url={https://github.com/snimu/blog/blob/main/contents/tokenization-and-batchnorm/README.md}
}
```
