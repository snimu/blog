# The output latent is for language and nothing else

The last layer's hidden state in a transformer is meant *only* for being decoded into token probabilities.

- *Don't* use it for autoregressive image generation
- *Dont't* use it for looped latent transformers
- *Only* use it to produce the next token in a language model

It is a compressed representation of the output probability distribution, one linear transformation + a softmax away from token space. It will always be in conflict with other tasks and modalities, because it is trained to collapse previous layers' abstract computations into a text-representation.

> The implication of this view is that the language head should be viewed as consisting of not only the single Fully Connected (FC) layer, but at least one prior transformer block, too.

My reasons for saying this will be wishy-washy and mostly intuition-lead. I still believe that the fundamental assertion is true, and I will illustrate it by showing clear benefits for multimodal models, and for latent looping.

## Multimodality

The [Chameleon paper](https://arxiv.org/abs/2405.09818) found that in early-fusion VLMs, the text- and image-modalities compete for resources in the attention mechanism. To prevent exploding attention weights, they make heavy use of QK-norms (as everybody should, but still), re-order all the norms inside the model, and sometimes use dropout.

But what if that conflict for resources is caused not by any inherent conflict between "thinking about" the modalities, but by the need for the model to provide very different information in the output latent between text and images, for it to be easy to decode them into the corresponding modality? If the output hidden states for text prediction need to look very different from those for image prediction, then throughout the model, gradient-updates on image-outputs will move the model to make the output-hidden-states fit for image prediction, and vice-versa. This obvisouly leads to conflict.

But if the last shared hidden states between the modalities are fed into another (single) transformer block for text, and a different one for images, then the two modalities will share only abstract thoughts, which can be made to blend information from different modalities much more easily, at least in my intuition.

A reason to believe that this might work is that DeepSeek's [Janus](https://arxiv.org/abs/2410.13848v1) and [Janus Pro](https://arxiv.org/abs/2501.17811) models explicitely de-couple image understanding from image generation, which already seems to help. Why not push this further and decouple abstract thinking from concrete instantiation into different modalities?

### Now do the same at the input

You've probably been screaming internally that everything I've said about the output hidden state is also applicable to the input. I would agree.

My intuition is that the first transformer block transforms the token-based input into a much more abstract representation.

So what if we performed SigLIP not into token-space, but into the abstract space after the first transformer block? Especially with continued training of both the text and image modalities, I imagine that this would allow the model to learn a much more compatible representation of both modalities, instead of forcing everything to be expressed in terms of text.

### The ideal multimodal model

My intuition for how a multimodal model should ideally look is:

- See the embedding layer *and the first transformer block* as the text-embedding layer
- Use separate mechanisms for image-understanding and image-generation, as in Janus and Janus Pro
- For both, project into the abstract space; both the SigLIP adapter and the VQ tokens (though maybe the typical fully-connected layer used as an adapter is enough if we project into the more abstract space, I would imagine that using a transformer block would help here, too)
- At the output, do the same thing: the language head consists of a transformer block *and* a fully-connected layer
- And the image-generation head also uses its own transformer block to decode the abstract hidden states into VQ tokens

There is one additional aspect that I would add: To aid image-understanding, I would make decode the images meant for image-understanding back into SigLIP-space and predict there. When doing this from a shared abstract hidden state, this should significantly aid image understanding. If we mask some input-tokens like in [MEAP](https://arxiv.org/abs/2502.07490), for both text and image-understanding, it would make for an even more capable model, but that's beside the point.

## Latent looping

In [COCONUT](https://arxiv.org/abs/2412.06769), the authors feed a language model's output hidden states back into its input, and found that this enables the model to perform Breadth-First Search over the tokens.

While this is advantageous for short bursts, it's still reasoning with explicit text, except that it's now a superposition of several text-strands. What people usually think is missing in LLMs is longer abstract thinking per token. If you want to produce more tokens, just produce more tokens.

Looping the hidden states from before the last transformer block into the hidden states after the first transformer block is actual latent reasoning. I don't know if it will work from a perspective of training dynamics, but if it does, it seems like a better way to loop transformer layers, which at minimum [saves costs](https://arxiv.org/abs/2412.06769).

## Citation

```bibtex
@misc{snimu2025multilayerlanguageheads,
    title={Multi-layer language heads: the output latent is for language and nothing else},
    author={Sebastian M\"uller},
    year={2025},
    month={mar},
    url={https://snimu.github.io/2025/03/30/multi-layer-language-heads.html}
}
```
