# Embeddings are in the middle of the model

*This is probably pretty obvious to many people, but was unintuitive to me. I'm quickly writing it down to not forget it, and to maybe help somebody.*

If you compare two LLMs that are identical except for the vocabulary size, the one with the larger vocabulary will obviously allow you to work with shorter sequences, because on average, it can group more characters into a single token. However, it also increases parameter count. I never understood why that should improve performance.

Here is what used to be my contention:

Most tokens are redundant because they are just re-combinations of other tokens. The token "token" is just a combination of five characters which are all tokens themselves. At the embedding layer, the parameters of these tokens never interact; why would more parameters be useful if they just encode re-combinations of each other?

The answer is that SGD bakes meaning about character-level interactions into the embedding-parameters. In the forward pass through the transformer, the tokens are mixed; consequently, the feedback signal (gradient) is also mixed in the backward pass. In this sense, the embedding parameters are at the end of a pass through the entire transformer, and can learn dependencies between each other.

They thus encode global statistics over the entire datset about the relationships between characters into tokens. A single embedding of a token consisting of five characters doesn't just encode those characters, but also the meaning of them in the order they appear in the token, as calculated by the entire transformer's backward pass.

In that sense, they are in the middle of the model.

(This is, of course, more of an analogy than a true equivalence; there are differences between input tokens being mixed in the forward pass, and target tokens being mixed in the backward pass, but it helps me understand what's going on.)

## Citation

```bibtex
@misc{snimu2024tokenmerge,
    title={Embeddings are in the middle of the model},
    author={Sebastian M\"uller},
    year={2024},
    month={dec},
    url={https://github.com/snimu/blog/blob/main/contents/embeddings-thoughts/article.md}
}
```
