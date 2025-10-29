# TODO: find title

In response to [my post](https://x.com/omouamoua/status/1976695893912174792) announcing [the article about my second modded-nanogpt medium world record](https://snimu.github.io/2025/10/10/modded-nanogpt-x0.html), several people made suggestions for variations of or alternatives to the technique introduced in the article.

I managed to test two of these: multiple embeddings per value embedding as suggested by [Braden Koszarsky](https://x.com/KoszarskyB), and variations in the per-layer embedding via linear transformations by [Danijar Hafner](https://x.com/danijarh). In this article, I will first quickly repeat the technique used in the previous article, and then go through the results for both of these suggested techniques.

## The baseline

TODO:

- multi-embedding
- value embeddings

My baseline incorporated a technique from [a subsequent modded-nanogpt record](https://snimu.github.io/2025/10/19/modded-nanogpt-backout.html) I made. The exact technique doesn't matter here, what's important is that since it is clearly composable with the multi-embedding technique, the results in this article should be meaningful.

## Multiple Embeddings per Value Embedding

[Braden Koszarsky](https://x.com/KoszarskyB) suggested to use
