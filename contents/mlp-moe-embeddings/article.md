# [Note to Self] MLP equals embedding layer, MoEs equals better one

> This is mostly a note to myself, containing information that is obvious to many but that I have just figured out. I publish it because forcing myself to publish does several things for me: 1) it forces me to actually write my thoughts down in a coherent manner, which 2) forces me to properly think through them, and 3) it allows me to maybe get some feedback from others on my Twitter [@omouamoua](https://x.com/omouamoua).

A Multi Layer Perceptron (MLP) is an embedding layer, and a Mixture of Experts (MoE) is a better one.

Table of Contents:

- [MLP equals embedding layer](#mlp-equals-embedding-layer)
- [MoE equals better embedding layer](#moe-equals-better-embedding-layer)
- [Why this matters](#why-this-matters)

## MLP equals embedding layer

A normal embedding layer is a collection of learned features that are activated by a fixed degree by different tokens and then summed; but the fixed activation is one for one token and zero for all others, so it can be efficiently implemented by simply picking a feature for each token instead of doing the matrix multiplication.

An MLP is a collection of learned features that are activated to different degrees by different tokens and then summed. Attention with positional embeddings is used to provide an order-dependent, weighted combination of tokens at the MLP input to activate the right features to the right degree before summing them up, which means that the number of possible feature combinations is much larger than the number of features; but it is still just a recall of learned features, and thus an embedding layer in the widest sense.

To summarize, both a normal embedding layer and an MLP can be viewed as embedding layers, but the former is activated very sparsely while the latter is activated in a dense manner.

## MoE equals better embedding layer

In terms of sparsity, an MoE is between an MLP and a typical embedding layer.

The router constrains the diversity of token-combinations that appear at the input of a single expert; the more total experts there are, the lower the diversity of inputs to each one should be. While the expert inputs typically aren't one-hot, they are similar in that in the limit&mdash;as we approach infinite experts&mdash;they should approach a fixed combination of learned features in the expert, which are then summed up in a fixed manner to produce a single fixed features, just like an embedding layer; then, each expert could be replaced with a single learned feature (at least during inference) that is activated in a binary manner by the routing layer.

Yes, we typically activate several experts at once and sum their results, and yes, we don't have infinite experts available to us, but increasing MoE sparsity makes an MLP more and more similar to a pure embedding layer.

## Why this matters

Consider the optimum for language modeling: have an embedding for every single possible token combination at the input (so `S_max^V` different embedding vectors, where `S_max` is the maximum sequence length and `V` is the vocabulary size), and decode it into the perfect probability distribution. This is just one gigantic embedding layer.

The reason for why this isn't done is that we simply don't have the data nor the compute to do that (and we never will, because `S_max^V` is ridiculously large for any desireable size of `S_max` (which increases as you decrease `V`, so it's enormous even for bytes)). So we instead train an embedding layer that approximates this: a transformer. It can produce more features at the output than it has features available, by performing a weighted sum over those features to produce unique, new features. This way, it can compress the optimal language model in a lossy manner, into something that can be learned from available data and with realistic compute resources. And despite all the things we add to make a transformer learnable (like stacking many layers, adding a residual, norming the activations, and so on), this is still a compression, because the number of possible sequences that require a unique output feature is just so huge.

In this viewpoint, Attention takes the role of selecting the right combination of features depending on the tokens in a sequence and their order. This is what allows the compression through a re-combination of features to work.

An MoE can have more features available for mixing than an MLP, at the same cost. Routing is just as dependent on the order-dependent token-mixing performed by Attention as the MLP; in fact, it performs the same operation as the MLP: feature selection. The reason we need an MLP at all instead of a simple FC layer plus a non-linearity (like the router) is that SGD-variants cannot learn the right features otherwise. So we have cheap experts (basically mini-MLPs) to learn good features that can be combined in a sensible manner within a restricted range of possible inputs, and a cheap router to select the right expert for each input.

In a way, this reminds me of linearization of non-linear control-systems: we first choose were on the full data-manifold we are, then use a (semi-)linear approximation of that part of the manifold to make a prediction. The router performs the former task, the experts the latter.
