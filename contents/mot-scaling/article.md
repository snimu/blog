# Scaling the Mixture of Tokenizers

The Mixture of Tokenizers (MoT) is an idea for combining different tokenizers.

In this case&mdash;the simplest but most important one&mdash;it is about mixing bytes and tokens at the input, and predicting bytes *or* tokens at the output. The aim is to combine the advantages of tokens and bytes, and results are looking promising.

I will give a proper motivation for every aspect of the architecture [below](#motivation). But first, I will explain the different components of the architecture and the choices that can be made for each, because otherwise, the motivation wouldn't make any sense.

## Architecture

### Tokens to Bytes

How did I extract bytes from tokens?

- Constant bpt
- Pad and cut off
- Input: left-pad -> pull in
- Output: right-pad -> pull out -> multi-byte (and effectively, multi-token prediction) -> see [sampling trajectories](#sampling-trajectories-from-multi-byte-predictions)
- Model sequence length `T_model` is the token sequence length `T_token`, not the byte sequence length `T_byte`

**Caveat:** I didn't actually use bytes. Instead, I used a vocabulary of all *characters* that appear in the entire GPT-2 token vocabulary. Since UTF-8 is made up of up to 4 bytes, there are more characters than bits in a byte; 458 characters in the case of GPT-2 vs. 256 unique bytes. I did that because it was easier to implement, and the numbers aren't off by much so my results should transfer. In fact, I believe that the vast, overwhelming majority of characters that are actually used in training can be represented by a single byte without overlap (most multi-byte characters are rather rare), so the likelihood that the results transfer is even higher. And having a vocabulary size of 256 compared to 458 has additional advantages, as we will see later. Having mentioned that, I'm going to speak about bytes again from now on, for the sake of simplicity.

### Byte Mixin

Here are the options that I've considered:

- `noop`: NoOp means tokens only. No bytes are mixed in.
- `concat`: For each token, concatenate the token embeddings and its constituent byte embeddings into a single vector, the project it into the model dimension `D_model` and use that as the final embeddings
- `cross_attn`: Perform cross-attention between the token and its constituent bytes. To make this cheap, perform the cross-attention only between the single token and its own bytes

Now, lets look at the two byte-mixin methods (ignoring `noop` because it doesn't mix in bytes) in detail.

#### Byte Mixin: Concat

*How does this work?*

We embed the tokens and their corresponding bytes independently. Then, we concatenate the byte embeddings to the token embeddings. If we keep the order of the token and bytes consistent, the Fully Connected layer we use afterwards to project into `D_model` can learn positional information.

*What advantages does this method have?*

Besides being simple, this method allows us to have three separate dimensions in the model:

- `D_token`: the embedding dimension of the tokens
- `D_byte`: the embedding dimension of the bytes
- `D_model`: the dimension of the residual stream

Token embeddings don't have to be huge to work well, so we can keep them small without it being a big loss. Evidence for this comes from embedding models: they compress multiple hidden state vectors of the model into a single embedding vector; often hundreds of vectors at once, reduced to a single one of the same dimension! And still, they retain semantic meaning pretty well. Another piece of evidence (for which I cannot find the citation, sorry) is that we can make models narrower and deeper, and they will work better! They're also more expensive due to the nature of GPUs, but it works. No, the large dimension of the residual stream is required for the actual transformation of the data, not its representation. Token embeddings are just data representations, so keeping them small is not an issue.

This will actually save us a lot of parameters. In GPT-2, the vocabulary size V is greater than 50.000, and most modern tokenizers exceed 100.000 or even 200.000 tokens. If we can reduce their embedding dimension from `V * D_model` to `V * D_token`, and make `D_token << D_model`, then we save *a lot* of parameters.

There are far fewer byte embeddings than token embeddings, so it doesn't matter as much here, but it is still advantageous that we can control `D_byte` independently of `D_token`. Because there are many bytes per input token, and they will all be concatenated to the token embeddings, it might make sense to make `D_byte < D_token` so that the token embeddings have more weight in the input.

#### Byte Mixin: Cross Attention

*How does this work?*

We independently embed the tokens and bytes and perform cross attention between the two sequences. Assuming we have 16 bytes per token (as is the case in the experiments presented below), we use an attention mask that allows attention between a token and the 16 bytes belonging to it (bytes pulled from earlier tokens might be among those, though). This can simply be implemented as a batch matrix multiplication with a softmax and some scaling.

*What advantages does this method have?*

It's attention-based which feels attractive. Theoretically, we could also relax the attention mask, which would give a larger receptive field from bytes to tokens. This could be advantageous, but since I see the byte mixin as part of the embedding process, I have very low confidence in that. I haven't tried it.

### Byte Mixout

We have `T_model` = `T_token` hidden states at the output of the model, but we want to predict `T_byte` bytes. How do we make that projection? I have come up with two options:

1. `copy`: Copy each hidden state `bpt` times (so 16 times in our case), then use a self-attention layer with a sliding-window attention mask to differentiate the bytes (see [below](#byte-self-attention)).
2. `split`: Split each output hidden state of dimension `D_model` into `bpt`  parts and re-arrange them into one continual sequence.

#### Byte Mixout: Copy

*How does it work?*

Again, just copy the hidden state belonging to a token `bpt` times and apply sliding-window self-attention with a residual. Models that work with tokens can obviously do the "mental work" of producing multiple bytes worth of material within a single tokens embedding, so this should work.

There is also evidence of it working, from when I tested it [on forward addition](https://x.com/omouamoua/status/1913657224548716608?s=46) (I haven't had the time to write it down more properly).

*What advantages does this method have?*

The biggest potential advantage is that it retains `D_model`. If we assume that it helps the language head to have a big latent to decode, then this would be good. And while having a tensor of size `B x 16*T_model x D_model` is memory-intensive, we are also decoding into the byte-vocabulary, a.k.a. 256 values (or 458 in my case), which saves much more memory in the language head than the copying costs. A linear layer that decodes from `D_model` into `V` is gigantic; if `V` is cut by a factor of 100 or more, that saves massive amounts of memory (and a decent amount of compute).

#### Byte Mixout: Split

*How does it work?*

Re-shape the output hidden states of the model from `B x T_model x D_model` into `B x T_model*16 x D_model/16`, assuming `bpt=16` and `T_model % bpt == 0`. We can either directly decode this re-shaped tensor into bytes, or run a self-attention layer over it.

*What advantages does this method have?*

It's simple, and it has literally zero cost beside the cost of a re-shaping operation. Since we decode the output hidden states into only 256 (or 458) values, we have now almost completely removed the cost of the language head.

### Byte Self-Attention

I optionally apply self attention to the bytes at the in- and/or output. These are represented by `n_layer_in` and `n_layer_out` (which is 0 if it's not explicitely stated).

This self-attention has a sliding window that is 128 bytes wide, so 8 tokens equivalent, which makes it very cheap despite the long sequence length of the bytes tensor. There are two options for how to shape the sliding window mask: 1) a normal, causal sliding window mask and 2) the same, but make the mask bidirectional within the bounds of a token so that the first byte of a token sees the last and the last one sees the first, they all see the bytes from previous tokens, but none of them see any bytes from future tokens. Since we still predict token-for-token, this isn't cheating, and it might improve the model a bit.

## Motivation

With the architecture out of the way, I can discuss why I believe each part is a good idea.

### Tokens vs. Bytes

In my Tokens vs Bytes (LINK) article, I've named the following points in favor of tokens:

- They reduce the sequence length
- They capture statistics about the training dataset

The former is obvious: a token is a group of bytes that often appear in human text in the given order, so we can reduce a whole group of bytes into a single token.

The latter is also a consequence of the fact that tokens are ordered groups of bytes. During training, their embeddings are trained to contain a lot of semantically relevant information about the specific ordered group of bytes that they are made up of, which removes the necessity for the transformer to perform that work during inference. Basically, the gradients of the model do work during training that the model itself won't have to do a test time, which frees up model capacity. I've previously written that Tokens are in the middle of the model (LINK), and during training, that is true.

I've also listed the followind disadvantages of tokens:

- They are not super legible
- The might incentivize memorization
- Undertrained tokens cause unpredictable behavior

The first point seems obvious: there is no way to know what components a token is made up of except by seeing it somewhere else in training: either it's spelled out explicitly using different tokens, or the model learns it via induction from mis-spellings. But legibility is important for many capabilities: doing math without access to the digits a number is made up of (as is often the case with tokenizers that group digits into sets of three) is incredibly difficult and requires memorization of a lot of facts. It works much worse than doing math on bytes, as I've shown here (LINK).

To compensate, models are forced to memorize. I believe that a model that relies heavily on memorization compared to generalization will carry this burden throughout training, and be more vulnerable to memorization in general. That's a big dose of speculation, though.

The last point is again well known: if a token's embeddings are for some reason not trained well, they can cause strange behaviors during inference; see SolidGoldMagiCarp (LINK). This is, again, an issue with legibility: among 256 bytes (or 458 symbols), every single one is going to be used very often, so there are no undertrained embeddings, and thus the model will be able to make sense of tokens that, when used as tokens instead of as a byte sequence, are undertrained.

### Why mix tokens and bytes at the input?

*Combining the advantages.*

When provided with both tokens and bytes at the input, the model can learn to make use of all the advantages of tokens and those of bytes, combining a short seuquence length with training-set wide statistics about a given arrangement of bytes and legibility.

*Adversarial robustness.*

Another advantage of providing both tokens and bytes at the input is that it provides a multi-resolution input for text. This was my original motivation: In Ensemble Everything Everywhere (LINK), the authors show for CNNs that providing the same image at multiple resolutions to a model's input improves its adversarial robustness significantly.

The best explanation for why this works that I've come across are J.D. Pressman's (LINK) Logos (LINK) tweets, which basically say that data implies its own generation process, and this tweet (LINK) by him where he applies the framework to Ensemble Everything Everywhere by pointing out that a process that produces an image that looks like the real thing in multiple resolutions is likely to *be* the real thing, at least more likely than something that only has to be convincing at a single resolution. And important detail here is that CNNs focus strongly on the highest frequency parts of an image, and tend to neglect the low-frequenccy parts. In a high-resolution image, it will thus focus on very high-frequency parts, in a low-resolution image on low frequency parts. Therefore, even if the high-resolution image contains all the data in the low-resolution image and more, it is effectively the case that providing the image at multiple resolutions enables the model to see it more completely.

Another perspective is that a multi-resolution image-input is similar to diffusion, which is spectral autoregression (LINK) (autoregression from low- to high-frequency parts of an image).

Combining tokens and bytes at the input will provide a similar effect, or at least I hope it does.

*What does that mean?*

My tentative view, formed from lots of very small experiments, thinking about the subject for long, and some intuition, is that the bytes will form the core of the input; they carry most of the information. There are two perspectives on what the token adds: 1) an information store about a specific combination of bytes, which is static but guaranteed to have been contextually relevant during training; 2) for special tokens, a steering vector.

### Why predict bytes at the output?

### Pulling bytes

- More context at input; contextualization machine
- Multi-token prediction at output

...

*Pulling bytes from previous tokens at the input.*

Pulling bytes from previous tokens provides redundancy and more context. I like the example given in Contextualization Machines (LINK), that a single token can quickly be turned into multiple tokens with a single mis-spelling. (PROVIDE CONCRETE EXAMPLE). With tokens, this pulls the model out of its learned distribution, but with pulled bytes, it will see the full word at the last of the three tokens, and be able to make the connection. In other words, pulling bytes can support an error correction algorithm.

And the model should still learn which bytes belong to the token and which ones don't, because the bytes that belong to the token will be the same no matter what, but the pulled bytes will differ based on context.

*Pulling bytes from future tokens at the targets.*

Multi-token prediction...

## Experiments

See code here...

## Results

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Calculated over last 100% of data

| mixin   | mixout   |   D_model |   D_tok |   D_byte | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| noop    | noop     |      1024 |    1024 |     1024 | 454.5M     |          312.11 |      3.64 |      4.22 |     1.19 |     1.60 |
| concat  | noop     |      1024 |     512 |       64 | 430.4M     |          313.08 |      3.62 |      4.12 |     1.18 |     1.57 |
| concat  | noop     |      1024 |     256 |       48 | 417M       |          309.60 |      3.61 |      4.08 |     1.17 |     1.49 |

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Calculated over last 50% of data

| mixin   | mixout   |   D_model |   D_tok |   D_byte | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| noop    | noop     |      1024 |    1024 |     1024 | 454.5M     |          312.11 |      3.11 |      4.22 |     0.38 |     1.60 |
| concat  | noop     |      1024 |     512 |       64 | 430.4M     |          313.08 |      2.99 |      4.12 |     0.20 |     1.57 |
| concat  | noop     |      1024 |     256 |       48 | 417M       |          309.60 |      3.06 |      4.08 |     0.22 |     1.49 |

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Calculated over last 10% of data

| mixin   | mixout   |   D_model |   D_tok |   D_byte | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| noop    | noop     |      1024 |    1024 |     1024 | 454.5M     |          312.11 |      3.00 |      4.22 |     0.19 |     1.60 |
| concat  | noop     |      1024 |     512 |       64 | 430.4M     |          313.08 |      2.91 |      4.12 |     0.09 |     1.57 |
| concat  | noop     |      1024 |     256 |       48 | 417M       |          309.60 |      2.93 |      4.08 |     0.09 |     1.49 |

## Future work

It's of course of interest to scale the models further and see how it goes, but my budget is limited so I want to focus on developing the technique itself further. I have two specific things in mind: finetuning existing, token-based models, and cheap decoding of multiple bytes at the output.

### Finetuning token-based models with MoT

### Sampling trajectories from multi-byte predictions
