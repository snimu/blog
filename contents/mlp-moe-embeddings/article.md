# [Note to Self] An MLP is an embedding layer, an MoE a sparse one

> This is mostly a note to myself, containing information that is obvious to many but that I have just figured out. I hope it still helps others.

A Multi Layer Perceptron (MLP) is an approximation of an embedding layer, and a Mixture of Experts (MoE) is a closer, sparser one.

Table of Contents:

- [MLPs are embedding layers](#mlps-are-embedding-layers)
- [MoEs are sparse embedding layers](#moes-are-sparse-embedding-layers)
- [Why this matters](#why-this-matters)
- [Summary](#summary)

## MLPs are embedding layers

A normal embedding layer is a collection of learned features that are activated to a fixed degree by different tokens and then summed; but the fixed activation is one for one token and zero for all others, so it can be efficiently implemented by simply picking a feature for each token instead of doing the matrix multiplication.

An MLP is a collection of learned features that are activated to different degrees by different tokens and then summed. Attention with positional embeddings is used to provide an order-dependent, weighted combination of tokens at the MLP input to activate the right features to the right degree before summing them up. This means that the number of possible feature combinations is much larger than the number of features; but it is still just a recall of learned features, and thus an embedding layer in the widest sense.

So while both a normal embedding layer and an MLP can be viewed as embedding layers, the former is activated very sparsely while the latter is activated in a dense manner.

## MoEs are sparse embedding layers

In terms of sparsity, an MoE is between an MLP and a typical embedding layer.

The router constrains the diversity of token-combinations that appear at the input of a single expert; the more total experts there are, the lower the diversity of inputs to each one should be. While the expert inputs typically aren't one-hot, they are similar in that in the limit&mdash;as we approach infinite experts&mdash;they should activate the expert's features in a fixed manner to produce a single fixed output feature per expert, just like an embedding layer; therefore, in the limit each expert could be replaced with a single learned feature that is activated in a binary manner by the routing layer.

> Why am I saying that they activate the experts in a fixed manner? Because the number of distinct inputs in a finite-precision, finite-size vector is itself finite. Therefore, when the number of experts exceeds the number of possible vectors at the input of the experts, each possible vector should ideally be routed to exactly one expert, and each expert should only ever be given a single vector as input.

Yes, we typically activate several experts at once and sum their results, and yes, we don't have infinite experts available to us, but increasing MoE sparsity makes an MLP more and more similar to a pure embedding layer.

## Why this matters

Consider the optimum for language modeling in terms of perplexity: having an embedding for every single possible token combination at the input (meaning a collection of `V ^ S_max` different embedding vectors, where `S_max` is the maximum sequence length and `V` is the vocabulary size), and decoding it into the perfect probability distribution. This is just one gigantic embedding layer.

The reason for why this isn't done is that we simply don't have the data nor the compute to do it (and we never will, because `V ^ S_max` is ridiculously large for any desireable value of `S_max`). So we instead train something that approximates this embedding layer: a transformer.

It can produce more features at the output than it has features available internally, by performing a weighted sum over those internal features to produce unique, new ones. This way, it can compress the optimal language model in a lossy manner, into something that can be learned from available data and with realistic compute resources. And despite all the things we add to make a transformer learnable in a practical setting (like stacking many layers, adding a residual, norming the activations, and so on) which add size and compute, this is still a compression, because the number of possible sequences that require a unique output feature is just so huge.

In this viewpoint, Attention takes the role of selecting the right combination of embeddings depending on the tokens in a sequence and their order. This is what allows the compression through a re-combination of features to work.

Crucially, an MoE can have more features than an MLP at the same cost. Routing is just as dependent on the order-dependent token-mixing performed by Attention as the MLP. In fact, it performs the same operation as the MLP: feature selection. The reason we need an MLP at all instead of a simple FC layer plus a non-linearity (like the router) is that SGD-variants cannot learn the right features otherwise. So we have cheap experts (basically mini-MLPs) to learn good features that can be combined in a sensible manner within a restricted range of possible inputs, and a cheap router to select the right expert for each input.

> Sidenote: This reminds me of linearization of non-linear control-systems: we first choose were on the full data-manifold we are, then use a (semi-)linear approximation of that part of the manifold to make a prediction. The router performs the former task, the experts the latter.

## Summary

This was a very roundabout way of saying "MoEs are more parameter-efficient than dense MLPs", but comparing both to normal embedding layers has helped me understand transformer more. I hope someone else can benefit from this, too.
