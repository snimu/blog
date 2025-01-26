# The output latent is for language and nothing else

The last layer's hidden state in a transformer&mdash;the activations that are decoded into token probabilities&mdash;is meant *only* for *that specific task*.

- *Don't* use it for autoregressive image generation
- *Dont't* use it for looped latent transformers
- *Only* use it to produce the next token in a language model

It is a compressed representation of the output probability distribution, one linear transformation + a softmax away from token space. It will always be in conflict with other tasks and modalities, because it is trained to collapse previous layers' abstract computations into a text-representation.

> The implication of this view is that the language head should be viewed as consisting of not only the single Fully Connected (FC) layer, but at least one prior transformer block, too.

## Multimodality

There are several reasons to make the language head deeper, and split outputs earlier.

### Chameleon

The authors of the [Chameleon paper](https://arxiv.org/abs/2405.09818) train a [Llama 2 model](https://arxiv.org/abs/2307.09288) to autoregressively predict not only text, but also images, in an early-fusion multimodal model. They state clearly that the different modalities compete for resources inside the model, and go through a lot of trouble to prevent exploding attention norms. They use QK-norms (as everybody should, but still), re-order all the norms inside the model, and sometimes use dropout.

But what if that conflict for resources is caused not by any inherent conflict between "thinking about" the modalities, but by the need for the model to provide very different information in the output latent between text and images, for it to be easy to decode them into the corresponding modality?

Then, simply splitting the stream of activations a few layers before the unembed layer rather directly prior to it would solve the issue that they discussed, or at least dampen it. I 

### Image understanding vs. generation

separation would also separate image understanding from image generation

#### Rant: the attention mask

(go into a little rant about the attention mask here)

- by producing all image-tokens at once with the new attention mask, you do more useful work (because you're not masking half the attention mask used in causal prediction); this might acutally prevent some problems where the model will produce some random but plausible looking shit at tokens 1 to 20, and at token 200 to 256 nothing fits together and it has to come up with plausible-looking but nonsensical solustions; with a bidirectional mask, the model can essentiall "plan" the entire image at once
- you can do this in a 256x more parallel fashion (assuming 256 tokens per image); which lends itself much more to MoEs during inference (see EpochAI again)

## looped latent transformers

more compatible because input and output work in token space, but the goal is to propagate abstract computations; these are partially lost.
