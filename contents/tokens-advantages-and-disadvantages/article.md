# Tokens vs. Bytes

Compared to bytes, tokens have two **advantages**: 1) They lead to shorter sequence lengths; 2) Their embeddings contain trainset-wide statistics on the specific combination of bytes that they consist of. They also have two **disadvantages**: 1) They are poorly legible; 2) They encourage memorization.

## Advantages

### Tokens lead to shorter sequence lengths

This point is obvious, and the main purpose of tokens. If you have a larger vocabulary, you are more likely to stumble upon tokens containing more bytes, which leads to shorter sequence lengths while covering the same information, which is very useful. Let's move on.

### Tokens contain trainset-wide statistics

I've written about this twice before: [Tokenization and batch-norm: incorporating global statistics](https://github.com/snimu/blog/blob/main/contents/tokenization-and-batchnorm/tokenization-and-batchnorm.md) and [Embeddings are in the middle of the model](https://github.com/snimu/blog/blob/main/contents/embeddings-thoughts/article.md).

Every token except the ones representing single bytes can be split into multiple shorter tokens of the same tokenizer. So beyond shortening sequence lengths, what's the point of having more tokens? The answer is that the embedding layer lies at the end of the backward pass through the entire transformer. Therefore, each embedding is updated with all other embeddings in mind, or at least all that are in context. This enables the embedding layer to learn global, trainset-wide statistics about the specific combination of bytes that each token consists of, relative to the other combinations of bytes (tokens) in the same sequence.

I want to stress how useful this is: simplifying a bit, instead of reading letter-by-letter, the models can now read word-by-word, and many of the common associations between the words are already known at the input to the transformer-proper.

In fact, even the [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) replicates this approach by dynamically adding n-gram embeddings to each byte embedding (see [my article on the Byte Latent Transformer](https://github.com/snimu/blog/blob/main/contents/byte-latent-transformer/article.md)).

## Disadvantages

### Tokens are poorly legible

What letters does the word "token" consist of? An LLM cannot possibly tell from the embedding of the token itself, because that doesn't give access to the bytes that make up the token. Instead, it has to memorize what bytes the token consists of.

This makes the following tasks more difficult (the list is likely not exhaustive):

- Math: doing any kind of math is extremely difficult if you don't have each and every digit in each of the numbers you're working with available at the input. Tokens often group three digits into one token. To perform many mathematical algorithms&mdash;like simple addition with a carry term&mdash;this requires models to internally recall the digits of the numbers, and *then* perform the addition.
- Character-level tasks like those from [the CUTE benchmark](https://arxiv.org/abs/2409.15452): "Spelling", "Inverse Spelling", "Contains", "Similarity", "Insertion", "Deletion", "Subsitution", and "Swapping".
- Rhyming and poetry. While the exact pronounciation of words cannot always be known even given their spelling (especially in English), knowing the spelling helps a lot. Again, tokens require memorizing the bytes of the words to achieve the same as a byte-level model.

### Tokens encourage memorization

Above, I've listed many tasks for which token-based models have to memorize the bytes in the tokens. On the one hand, this makes those tasks harder even if the bytes are successfully memorized, because the model has to recall the bytes internally for the tasks. On the other hand, it also encourages the models to learn by memorization.

Imagine that the trainset contains three texts that say essentially the same thing, but with subtly different wording. To learn good predictions for the third text from the first two efficiently, the model can follow two strategies: 1) generalize from the first two texts, recognize that the third text is about the same thing, and produce the correct tokens from context; 2) memorize the similarity between the first two texts, and that one wording is very similar to the other in meaning, so that when the text appears a third time with slightly different wording again, the model can recognize the similarity and simply produce snippets from one of the previous texts.

If a model works with tokens, it *must* first memorize which tokens mean the same thing; "t" "o" "k" "e" "n" is the same as "token". This is because if the same text is written slightly differently, the same words can be tokenized differently. If the model is forced to memorize the tokens already, it will be more likely to also memorize other patterns.

I believe that what the model learns early in training strongly determines the entire training run. Therefore, encouraging memorization early on is like consciously picking an anti-lottery-ticket initialization at the start of training!

## Mixture of Tokenizers

In this context, I need to mention the [Mixture of Tokenizers](https://github.com/snimu/blog/blob/main/contents/fixing-tokenization/README.md) (MoT), because the advantages and disadvantages of tokens map perfectly onto it.

Background: The simplest Mixture of Tokenizers enriches the token embeddings with byte embeddings through cross-attention. You embed the bytes (and pad them so that they correspond to the tokens), apply self-attention with a narrow sliding window to the bytes only, then apply cross-attention between a single token and the corresponding byte embeddings (this can also be replaced by a batch matrix multiplication).

This way, we get both the advantages of tokens: shorter sequence lengths, and dataset-wide statistics. We also avoid their disadvantages: the bytes of which a token consists are directly available to the model and legible; and this also prevents the memorization effect I hypothesized above.

And early results show clearly that the Mixture of Tokenizers are at a minimum vastly better at math than pure token-based models, and not worse at other tasks. This was shown by both [@nickcdryan](https://x.com/nickcdryan) [here](https://x.com/nickcdryan/status/1884298932001595529), and by myself [here](https://github.com/snimu/blog/blob/main/contents/mixture-of-tokenizers-math/article.md). I suspect that (and will test whether) this will also be great for reasoners.

## Citation

```bibtex
@misc{snimu2025tokensvsbytes,
    title={Tokens vs. Bytes},
    author={Sebastian Nicolas Müller},
    year={2025},
    month={3},
    url={https://github.com/snimu/blog/blob/main/contents/tokens-advantages-and-disadvantages/article.md}
}
```
