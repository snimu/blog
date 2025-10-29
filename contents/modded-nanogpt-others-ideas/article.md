# TODO: find title

In response to [my post](https://x.com/omouamoua/status/1976695893912174792) announcing [the article about my second modded-nanogpt medium world record](https://snimu.github.io/2025/10/10/modded-nanogpt-x0.html), several people made suggestions for variations of or alternatives to the technique introduced in the article.

I managed to test two of these: multiple embeddings per value embedding as suggested by [Braden Koszarsky](https://x.com/KoszarskyB), and variations in the per-layer embedding via linear transformations by [Danijar Hafner](https://x.com/danijarh). In this article, I will first quickly repeat the technique used in the previous article, and then go through the results for both of these suggested techniques.

## The baseline

TODO:

- multi-embedding
- value embeddings

My baseline incorporated a technique from [a subsequent modded-nanogpt record](https://snimu.github.io/2025/10/19/modded-nanogpt-backout.html) I made. The exact technique doesn't matter here, what's important is that since it is clearly composable with the multi-embedding technique, the results in this article should be meaningful.

## Multiple Embeddings per Value Embedding

[Braden Koszarsky](https://x.com/KoszarskyB) suggested to use multiple embeddings per value embedding. To test out the effect of this, I varied two things:

1. The number of `nn.Embedding` modules (I'll call those "Embedding Modules") per Value Embedding
2. The number of Value Embeddings

Here's how a Value Embedding with three Embedding Modules is created:

```python
ve: list[nn.Embedding] = ...

# Produce the values as usual from data:
q, k, v = F.linear(x, W_qkv).chunk(3, dim=-2)  # simplified

# Bias the values with multiple learned embeddings,
# in a weighted sum.
# The scalar lambdas l1-l4 are also learned during training.
value_embedding = l1 * ve[0] + l2 * ve[1] + l3 * ve[2]
v = l4 * v + value_embedding

# Apply Attention, etc.
...
```

I preserved the structure of the baseline, where each Value Embedding (now made up of multiple Embedding Modules) is applied to two layers: if there is one Value Embedding, it's applied to layers 0 and 15, if there are two then the first is applied to layers 0 and 14 and the second to layers 1 and 15, and so on.

While the Embedding Modules stay the same between the two applications of the Value Embedding, the learned lambdas differ, and we thus allow the model to learn different effective Embeddings between the two layers that each Value Embedding is applied to.

For now, let's ignore the wallclock time, and just see if the additional Embedding Modules improve performance per training step at all. If they don't&mdash;if, unlike in the article this is based on, adding more Embedding Modules *doesn't* consistently improve per-step performance&mdash;then this is pointless anyway. So here is a heatmap of the final validation loss for training runs using everything from one to four Value Embeddings, each using anything from one to five Embedding Layers:

![Heatmap final val loss](images/heatmap_val_loss_final.png)

There are two observations to be made here:

1. Increasing the number of Value Embeddings consistently lowers loss
2. Increasing the number of Embedding Modules per Value Embedding tends to decrease the loss, but not fully consistently

To better see these two trends, I have created two variations of the same heatmap. The first normalizes along the rows: for each row, show the loss as a percentage of the maximum loss in that row. This nicely shows how the number of Embedding Modules per Value Embedding impacts the final validation loss, for each number of Value Embeddings:

![Heatmap final val loss as percent of row-max](images/heatmap_val_loss_final_row_percent.png)

TODO

...

Surprising that more val embs means that more embs per val emb works better; each is still only applied to two layers. Maybe having an individual embedding at each layer only makes sense if there are enough layers, but if there are too few, tying them just leads to better performance?

...

TODO: col-max
