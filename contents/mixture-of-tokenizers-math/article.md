# Mixture of Tokenizers &mdash; Performance on addition and modular addition

[Mixtures of Tokenizers](../mixture-of-tokenizers/README.md) (MoT) make learning math easier than using classical tokenizers.

In this article, I will show this with several ablations.

You can find the code [here](https://github.com/snimu/mixture-of-tokenizers/tree/main/mathblations). Under *ablations.sh*, you can find the ablation commands I ran.

## Illustration

First, the following image gives an overview of the method:

![MoT for math](images/mot-math.png)

Now for the details.

## Model architecture

I trained two models on math data:

- **The MoT** &mdash; A normal transformer, but the token embeddings are enriched with digit-level embeddings through cross-attention. The cross-attention is masked so that each token only sees the digits that make it up. An attention layer is applied to the digits before the cross-attention to mix the digits. It is causally masked. In a larger, real model, I would use sliding-window attention here, but the context length is so short that is was not worth the implementation effort (however low it is).
- **The baseline** &mdash; A normal transformer. To make up for the additional parameters in the digit embeddings, the attention layer applied to the digits, and the cross-attention, I added one more transformer block to the baseline. Because that block includes a MLP, the baseline always has slightly more parameters than the MoT.

The models are modified from [this speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/102024_ScaleUp1B/c0078066-c8c9-49c8-868a-ff4d4f32e615.txt) by [KellerJordan](https://github.com/KellerJordan). All norms are RMS-norms without weights. They use QK-norms, and layer-tying between the embedding layer and the language head. Otherwise, they have a very standard architecture.

The speedrun used [the Muon optimizer](https://github.com/KellerJordan/Muon), and so do I. The number of training steps is adapted to the difficulty of the task.

## Data

All data takes the form `f(x1, x2) = y`. For example, when using addition, a sequence might look like this: `15 + 368 = 383`.

I ablate over two variables:

- **Maximum digits per token (dpt)** &mdash; The maximum number of digits that can be in a single token. If this number is $3$, then we have $1000$ tokens, plus the addition and equality tokens (and a padding token to make all sequences the same length, for practical purposes).
- **Maximum tokens per number (tpn)** &mdash; The maximum number of tokens that can be in a single number. For example, if `dpt=3` and `tpn=2`, then we can represent all numbers from $0$ to $999,999$. A number like $1,000$ would be represented as `[100, 0]`; $3$ as `[3]`, and so on.

Importantly, these variables relate the `x1` and `x2`, not `y`; `y` can be larger, of course.

I turn the tokens into digits by simply splitting their digits up. I turn each token into `dpt` entries. If `dpt=3`, and the number is $123$, then that number would be represented as `[1, 2, 3]`. $5$ would be `[<pad>, <pad>, 5]`, and so on. The padding always comes before the digits (though this doesn't matter as long as it is consistent). Operating tokens &mdash; "+" and "=" for addition &mdash; are similarly padded to `[<pad>, <pad>, +]` and `[<pad>, <pad>, =]`.

I run ablation and modular addition (mod 23, for no particular reason) and vary `dpt` and `tpn` over the following ranges: Every combination of `dpt` $\in [2, 4]$ and `tpn` $\in [1, 3]$. I run every setting five times to get statistically significant results.

## Results

...
