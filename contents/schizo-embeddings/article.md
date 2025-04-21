# Schizo embeddings: Initializing special tokens the complicated way

I've read some [complaints about Llama using too many special tokens](https://x.com/kalomaze/status/1908603782193103017?s=46). One of the issues with this is that those tokens are typically initialized randomly and then trained for a very short time in post-training, which leads to poorly trained embeddings and weird behavior.

A better approach is used by [Pleias](https://pleias.fr/), as described [by Alexander Doria here](https://x.com/dorialexander/status/1908621506546196666?s=46): take the last token in the tokenizer vocabulary as the initial value for the special tokens. That embedding is typically trained enough to at least not be completely out of place with the other embeddings, but sufficiently little that it doesn't bias your special tokens too much, allowing them to be trained meaningfully in just post-training.

Another approach is to [initialize the special tokens with the average of the other embeddings](https://nlp.stanford.edu/~johnhew/vocab-expansion.html). This obviously also moves it into the existing embedding space. However, [PaliGemma](http://arxiv.org/abs/2407.07726) tried this method and compared it to random initilization, and found that random initialization was better. I suspect that this is because being the average of the embeddings is extremely meaningless, and biases the embeddings strongly against learning whatever the special tokens are supposed to be.

Here is an approach that is so much less elegant than Pleias's that I call it Schizo embeddings, but that holds promise in my eyes.

## Special tokens have meaning

The starting point is that special tokens have meaning. Normal tokens have meaning, too. So if we can express the meaning of the special tokens in terms of the existing tokens (which we obviously can), then we might be able to find the token that is the closest to the special token's meaning among the entire vocabulary and initialize the special token with its embedding. Of course, this match would have to be done for each special token, not all of them at once.

I believe that this would not cause the same issue as using the average of all embeddings, because the initilization would be very explicitly towards a specific meaning. It would also allow the model to somewhat understand the purpose of the special token from the beginning of post-training, which would enable it to learn other things earlier on. And the special token would certainly not be undertrained.

Two questions remain:

1. How do we make sure that initializing the special token with the single closest match doesn't bias it too much?
2. How do we find the closest match?

## Preventing bias

Initializing a special token to the embeddings of a highly trained token from pre-training might overly bias the special token towards that token's meaning. Additionally, if this embedding is now used in a different context than during pre-training, adapting the model to the special context might change its behavior towards the original token.

To prevent these issues, it would likely be enough to simply take the mean of the k best-fitting token-embeddings, for example the best 5 or 10. The average of 10 tokens can express much more complex concepts than a single token can, and it is sufficiently different from any single token that the model can learn a behavior only for that specific combination of tokens, without affecting its behavior towards any of the individual tokens that make up the initialization overly much.

## Finding the best-fitting tokens

Sure, a human can sift through the entire vocabulary and write down what they believe is the best-fitting token for each special token, but that is a tedious and error-prone task. What are similar tokens to the model might differ from what are similar tokens to humans.

A simpler way would be to write a concise description of the special token's meaning in a sentence or two, then embed that description and every individual token in the vocabulary, and find the closest match to the description. Theoretically, we could use any embedding model here, though of course using one derived from the model we want to post-train would be ideal.

This would be a very simple process, yet one likely to yield good results.

### Using the model itself

An alternative to using an embedding model to find the similarity to the description is to use the model itself. Forward pass every single token in the vocabulary and save the hidden states halfway in the model layers, or at least one or two layers before the output layer. There, the embeddings are in their most abstract representation. If we also forward pass the description, save the hidden states at the same layer, and pool them (for example by taking the mean, or the exponential moving average at the last token position), we can find the similarity scores of each token to the description in the model's own abstract representation, and continue [with the next step](#pooling-the-embeddings).

A potential additional ablation to run would be to simply use the hidden states of the description as the initialization of the special tokens. Those woud be highly meaningful and abstract, but potentially too far away from the embedding space of the input layer. It might work if we use the hidden states after applying the first one or two layers of the model, but I'm not sure. If it does work, this is likely the best approach, because it would be guided by the model itself, very close the the embedding layer but still in a slightly abstract space.

Of course, all of these only work if people can be bothered to write an actually meaningful description for the special tokens.

## Pooling the embeddings

I've previously said that we should take the mean of the k best-fitting token-embeddings. However, some tokens might contribute more to the special token's meaning than others. How do we combine them?

The obvious solution is to take a weighted sum over the chosen embeddings, and use the similarity score of each embedding to the description as the weight for that embedding (though we might need to normalize the weights to sum to 1).

Another approach would be to use those same weights as initial values, but then perform one or two update steps in post-training to update only the weights for the sum of the special tokens. I doubt that it would be worth the trouble, but I'm ready to be positively surprised. And if it does lead to improved results to perform one such update step, it would be cheaper than updating all model weights.

## Conclusion

I've effectively presented two approaches to initialize special tokens: the main one of embedding a description and the vocabulary, finding the top-k closest tokens, and pooling them, optionally with a weighting based on the similarity to the description; and a second approach of simply using the hidden states of the first or second model layer of the description as the initialization of the special tokens.

These ideas are much more invovled than the common approaches of random initilization, initialization from the mean of all embeddings, or initialization from the embeddings of the last token in the vocabulary, but they have the potential to be better.

The approach would only be worthwhile if there is significantly less post-training then pre-training data, and the random-initilization-baseline is still a strong contender for the special token initialization.

## Citation

```bibtex
@misc{snimu2025schizoembeddings,
    title={Schizo embeddings: Initializing special tokens the complicated way},
    author={Sebastian M\"uller},
    year={2025},
    month={04},
    url={https://snimu.github.io/2025/04/21/schizo-embeddings.html}
}
```
