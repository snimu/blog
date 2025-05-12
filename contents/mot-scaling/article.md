# Scaling the Mixture of Tokenizers

The Mixture of Tokenizers (MoT) is an idea for combining different tokenizers.

In this case&mdash;the simplest but most important one&mdash;it is about mixing bytes and tokens at the input, and predicting bytes *or* tokens at the output. The aim is to combine the advantages of tokens and bytes, and results are looking promising.

I will give a proper motivation for every aspect of the architecture [below](#motivation). But first, I will explain the different components of the architecture and the choices that can be made for each, because otherwise, the motivation wouldn't make any sense. Then, I will present the highly promising [results](#results), and finish with [future work](#future-work).

Table of Contents:

- [**Architecture**](#architecture)
  - [*Tokens to Bytes*](#tokens-to-bytes)
  - [*Byte Mixin*](#byte-mixin)
    - [Byte Mixin: Concat](#byte-mixin-concat)
    - [Byte Mixin: Cross Attention](#byte-mixin-cross-attention)
  - [*Byte Mixout*](#byte-mixout)
    - [Byte Mixout: Copy](#byte-mixout-copy)
    - [Byte Mixout: Split](#byte-mixout-split)
    - [Byte Mixout: Split and Copy](#byte-mixout-split-and-copy)
  - [*Byte Self-Attention*](#byte-self-attention)
- [**Motivation**](#motivation)
  - [*Tokens vs. Bytes*](#tokens-vs-bytes)
  - [*Why mix tokens and bytes at the input?*](#why-mix-tokens-and-bytes-at-the-input)
  - [*Why predict bytes at the output?*](#why-predict-bytes-at-the-output)
  - [*Pulling bytes*](#pulling-bytes)
- [**Experiments**](#experiments)
- [**Results**](#results)
  - [*1000 steps (~500M tokens)*](#1000-steps-500m-tokens)
  - [*10,000 steps (~5B tokens)*](#10000-steps-5b-tokens)
  - [*100,000 steps (~50B tokens)*](#100000-steps-50b-tokens)
- [**Future Work**](#future-work)
  - [*Finetuning token-based models with MoT*](#finetuning-token-based-models-with-mot)
  - [*Sampling trajectories from multi-byte predictions*](#sampling-trajectories-from-multi-byte-predictions)
  - [*Comparing to a more modern tokenizer*](#comparing-to-a-more-modern-tokenizer)
  - [*Longer byte sequences*](#longer-byte-sequences)

## Architecture

### Tokens to Bytes

*How did I extract bytes from tokens?*

The trick for enabeling a mixing of tokens and bytes is to have a constant number of bytes per token (`bpt`). I achieve this by padding tokens with fewer bytes than my chosen `bpt` with a special padding byte, and cut off the last bytes of tokens with more than my chosen `bpt`.

In my case, I chose `bpt=16`. The longest token in the GPT-2 tokenizer has 66 bytes, there is no relevant token longer than 21 bytes, and very few over 16 bytes. Since I had a much cruder method for mixing tokens and bytes back then, I chose the lowest number of `bpt` that I felt I could get away with. I was especially worried that if I chose the full 66 bytes per token, almost all byte sequences would consist almost exclusively of padding bytes (the average token has ~6 bytes) and I felt that that would reduce performance significantly. However, if I did it again, I would at least try to increase `bpt`, but I will come to that [later](#longer-byte-sequences).

An important reason why I believe that more bytes per token wouldn't just give a more faithful representation of the tokens, but more general advantages, is that I also "pull the byptes".

For bytes at the input, I put the padding bytes to the left of the bytes that make up the token ("left padding"). Then, at token position `i`, I pull in the bytes from the previous tokens, until all padding bytes are filled or there are no more tokens to pull from.

At the output, I use right-padded byte sequences as targets (if I predict bytes instead of tokens), and pull bytes from future tokens.

Here's an example of how that might look for `bpt=6`; ":" is short for a padding byte and "_" for a space. Each token (and the bytes belonging to a token) is marked by square brackets.

```cmd
RAW:

Input Tokens    |    [Hi]     [,]      [_my]    [_name]  [_is]
Input Bytes     |    [::::Hi] [:::::,] [:::_my] [:_name] [:::_is]
Target Bytes    |    [:::::,] [_my:::] [_name:] [_is:::] [_snimu]

---

PULLED:

Input Tokens    |    [Hi]     [,]      [_my]    [_name]  [_is]
Input Bytes     |    [::::Hi] [:::Hi,] [Hi,_my] [y_name] [ame_is]
Target Bytes    |    [,_my_n] [_my_na] [_name_] [_is_sn] [_snimu]
```

While that looks confusing to humans, remember two things:

1. In reality, `bpt=16` (and more in the future), so there is much more context per byte
2. We're not training humans. What counts is the information content we give the model at the input, and the information content we make it predict in the targets

I will go more deeply into the advantages of doing this [below](#pulling-bytes).

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

#### Byte Mixout: Split and Copy

This is purely speculation and I have not tested it, but it should obviously be possible to combine `split` and `copy`: just split the output latents `bpt / c` times, then copy each latent that was achieved this way `c` times.

This might be useful if `bpt` is very high compared to `D_model`. If `bpt` is high, `copy` is expensive, because we get an output shape `B x (T * bpt) x D_model`. On the other hand, `split` gives the shape `B x (T * bpt) x (D_model / bpt)`, meaning that there as many elements in the output latents as with `noop` as the byte-mixout method. However, if `D_model` is small and `bpt` is high, this might lead to very small latents per byte that is decoded, and thus to a poor probability distribution over the bytes.

A combination of both could be a good compromise.

### Byte Self-Attention

I optionally apply self attention to the bytes at the in- and/or output. These are represented by `n_layer_in` and `n_layer_out` (which is 0 if it's not explicitely stated).

This self-attention has a sliding window that is 128 bytes wide, so 8 tokens equivalent, which makes it very cheap despite the long sequence length of the bytes tensor. There are two options for how to shape the sliding window mask: 1) a normal, causal sliding window mask and 2) the same, but make the mask bidirectional within the bounds of a token so that the first byte of a token sees the last and the last one sees the first, they all see the bytes from previous tokens, but none of them see any bytes from future tokens. Since we still predict token-for-token, this isn't cheating, and it might improve the model a bit.

## Motivation

With the architecture out of the way, I can discuss why I believe each part is a good idea.

### Tokens vs. Bytes

In my [Tokens vs. Bytes](https://snimu.github.io/2025/03/07/tokens-vs-bytes.html) article, I've named the following points in favor of tokens:

- They reduce the sequence length
- They capture statistics about the training dataset

The former is obvious: a token is a group of bytes that often appear in human text in the given order, so we can reduce a whole group of bytes into a single token.

The latter is also a consequence of the fact that tokens are ordered groups of bytes. During training, their embeddings are trained to contain a lot of semantically relevant information about the specific ordered group of bytes that they are made up of, which removes the necessity for the transformer to perform that work during inference. Basically, the gradients of the model do work during training that the model itself won't have to do a test time, which frees up model capacity. I've previously written that [Embeddings are in the middle of the model](https://snimu.github.io/2024/12/24/embeddings-are-in-the-middle-of-the-model.html), and during training, that is true.

I've also listed the followind disadvantages of tokens:

- They are not super legible
- The might incentivize memorization
- Undertrained tokens cause unpredictable behavior

The first point seems obvious: there is no way to know what components a token is made up of except by seeing it somewhere else in training: either it's spelled out explicitly using different tokens, or the model learns it via induction from mis-spellings. But legibility is important for many capabilities: doing math without access to the digits a number is made up of (as is often the case with tokenizers that group digits into sets of three) is incredibly difficult and requires memorization of a lot of facts. It works much worse than doing math on bytes, as I've shown [here](https://snimu.github.io/2025/01/28/mixture-of-tokenizers-math.html).

To compensate, models are forced to memorize. I believe that a model that relies heavily on memorization compared to generalization will carry this burden throughout training, and be more vulnerable to memorization in general. That's a big dose of speculation, though.

The last point is again well known: if a token's embeddings are for some reason not trained well, they can cause strange behaviors during inference; see [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation). This is, again, an issue with legibility: among 256 bytes (or 458 symbols), every single one is going to be used very often, so there are no undertrained embeddings, and thus the model will be able to make sense of tokens that, when used as tokens instead of as a byte sequence, are undertrained.

### Why mix tokens and bytes at the input?

*Combining the advantages.*

When provided with both tokens and bytes at the input, the model can learn to make use of all the advantages of tokens and those of bytes, combining a short seuquence length with training-set wide statistics about a given arrangement of bytes and legibility.

*Adversarial robustness.*

Another advantage of providing both tokens and bytes at the input is that it provides a multi-resolution input for text. This was my original motivation: In [Ensemble Everything Everywhere](https://arxiv.org/abs/2408.05446), the authors show for CNNs that providing the same image at multiple resolutions to a model's input improves its adversarial robustness significantly.

The best explanation for why this works that I've come across are [J.D. Pressman](https://x.com/jd_pressman)'s [Logos tweets](https://x.com/jd_pressman/status/1918419540788297856), which basically say that data implies its own generation process, and [this tweet](https://x.com/jd_pressman/status/1856866399920295955?s=46) by him where he applies the framework to Ensemble Everything Everywhere by pointing out that a process that produces an image that looks like the real thing in multiple resolutions is likely to *be* the real thing, at least more likely than something that only has to be convincing at a single resolution. And important detail here is that CNNs focus strongly on the highest frequency parts of an image, and tend to neglect the low-frequenccy parts. In a high-resolution image, it will thus focus on very high-frequency parts, in a low-resolution image on low frequency parts. Therefore, even if the high-resolution image contains all the data in the low-resolution image and more, it is effectively the case that providing the image at multiple resolutions enables the model to see it more completely.

Another perspective is that a multi-resolution image-input is similar to diffusion, [which is spectral autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html) (autoregression from low- to high-frequency parts of an image).

Combining tokens and bytes at the input will provide a similar effect, or at least I hope it does.

*What does that mean?*

My tentative view, formed from lots of very small experiments, thinking about the subject for long, and some intuition, is that the bytes will form the core of the input; they carry most of the information. There are two perspectives on what the token adds: 1) an information store about a specific combination of bytes, which is static but guaranteed to have been contextually relevant during training; 2) for special tokens, a steering vector.

### Why predict bytes at the output?

### Pulling bytes

*Pulling bytes from previous tokens at the input.*

Pulling bytes from previous tokens provides redundancy and more context. I like the example given in [Contextualization Machines](https://stochasm.blog/posts/contextualization-machines/), that a single token can quickly be turned into multiple tokens with a single mis-spelling: "hello" is a single token, but "hwllp" is three: "hw", "ll", and "p". With tokens, this pulls the model out of its learned distribution, because there is less context directly at the input, and the model will have to re-construct that context during its forward pass. But with pulled bytes, it will see the full word at the last of the three tokens (plus more context), and be able to make the connection. In other words, pulling bytes can support an error correction algorithm.

And the model should still learn which bytes belong to the token and which ones don't, because the bytes that belong to the token will be the same no matter what, but the pulled bytes will differ based on context.

*Pulling bytes from future tokens at the targets.*

If the average token consists of 6 bytes (its a bit more in the GPT-2 tokenizer, and even more in other tokenizers, but it's approximately true), and we predict 16 bytes for each token, then we predict on average 2.6 tokens into the future. The [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437) showed clearly that multi-token prediction (MTP) is very adventageous. By pulling bytes, we can get this for free.

## Experiments

The code is [here](https://github.com/snimu/mixture-of-tokenizers) under "scaled-pre-train".

## Results

The validation losses between models that predict tokens and those that predict bytes aren't directly comparable. After all, CrossEntropyLoss depends on the sum over the exponentials of all predicted classes (tokens or bytes). If the number of predicted classes changes, then the losses aren't truly comparable anymore, and since there are far fewer bytes than tokens (by more than 100x), this issue applies to comparing `noop` and `split`/`copy` mixout methods.

Therefore, whenever I'm comparing losses, I will be comparing different byte-mixin methods with `noop` as mixout, and different byte-mixout methods (that are not `noop`) using the same byte-mixin method to each other. I will compute the mean

I will also present evaluation results. (TODO: do the actual evals).

To save myself some money, I first ran a lot of experiments training for 1000 steps (~500M tokens), then fewer for 10,000 steps (~5B tokens), and, finally, the most promising experiments on 100,000 steps (~50B tokens).

### 1000 steps (~500M tokens)

(TODO: write down results)

### 10,000 steps (~5B tokens)

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Statistics calculated over last 10% of loss curve.

| mixin   | mixout   |   D_model |   D_tok |   D_byte | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| noop    | noop     |      1024 |    1024 |     1024 | 454.5M     |          312.11 |      3.00 |      4.22 |     0.19 |     1.60 |
| concat  | noop     |      1024 |     512 |       64 | 430.4M     |          313.08 |      2.91 |      4.12 |     0.09 |     1.57 |
| concat  | noop     |      1024 |     256 |       48 | 417M       |          309.60 |      2.93 |      4.08 |     0.09 |     1.49 |

### 100,000 steps (~50B tokens)

Let's first compare the MoT with `concat` at the input to the token baseline:

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Statistics calculated over last 10% of loss curve.

| mixin   | mixout   |   D_model |   D_tok |   D_byte | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| noop    | noop     |      1024 |    1024 |     1024 | 454.5M     |          300.95 |      5.46 |      5.66 |     0.19 |     0.66 |
| concat  | noop     |      1024 |     256 |       48 | 417M       |          296.07 |      4.21 |      4.82 |     0.06 |     0.80 |

Clearly, the MoT achieves significantly better results than the token-baseline, and does so with fewer parameters and a lower per-step time, in two ways:

1. The mean loss over the last 10% of training is significantly lower in both fineweb and finemath. This is an obvious advantage
2. The standard variation in the validation losses is much lower for the MoT than for the baseline. This indicates to me that the MoT is less prone to overfitting: a high standard deviation shows that that loss goes up and down rapidly. My main explanation for this effect is that the training data contains a bunch of sequences that are very similar to the validation data in one step, and the baseline overfits to that, decreasing the validation loss; then in the next batch, the sequences between the two are much more dissimilar, and the validation loss increases again. The MoT on the other hand shows much lower variation between the two scenarios, because it genralizes better. Note that I haven't explicitly tested this theory.

Now, let's compare the losses between the two byte-mixout methods:

*fw*: validation loss on fineweb data;
*fm*: validation loss on finemath data.
Statistics calculated over last 10% of loss curve.

| mixin   | mixout   |   D_model |   D_tok |   D_byte |   # layers out | # params   |   step time [s] |   mean fw |   mean fm |   std fw |   std fm |
|:--------|:---------|----------:|--------:|---------:|---------------:|:-----------|----------------:|----------:|----------:|---------:|---------:|
| concat  | split    |      1024 |     256 |       48 |              0 | 365.5M     |          258.13 |      5.47 |      5.45 |     0.17 |     0.25 |
| concat  | copy     |      1024 |     256 |       48 |              1 | 366M       |          262.88 |      6.06 |      5.57 |     0.35 |     0.50 |

It seems like `split` is both cheaper and better than `copy`, which is nice.

## Future work

It's of course of interest to scale the models further and see how it goes, but my budget is limited so I want to focus on developing the technique itself further. I have two specific things in mind: finetuning existing, token-based models, and cheap decoding of multiple bytes at the output. Additionally, I think that testing higher `bpt` would be worthwhile, as would comparing to a more modern tokenizer such as Qwen's.

### Finetuning token-based models with MoT

See [original MoT post](https://x.com/jd_pressman/status/1856866399920295955?s=46)

### Sampling trajectories from multi-byte predictions

### Comparing to a more modern tokenizer

The great advantage that the MoT seems to have over the baseline in math performance is likely in large part an effect of the GPT-2 tokenizer, which tokenizes numbers in groups of up to three digits per token, which is algorithmically disadvantageous. More modern tokenizers like the one by Qwen often keep every single digit as its own token, which likely negates that MoT advantage. However, all the other advantages of the MoT that I've listed above should still apply (and I have collected them in [this tweet](https://x.com/omouamoua/status/1921906477154840749), if you're interested).

If time- and money-related constraints allow for it, I will test this out.

### Longer byte sequences

Pulling bytes at input -> longer sequences will only give more context (though in the first few bytes, there will still be plenty of padding, now more than ever).

Pulling bytes at output -> multi-byte / multi-token prediction -> would be intereting to take this further (and at some point, reducing the loss weights for the far-out bytes is an option, too)
