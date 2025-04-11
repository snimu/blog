# My dream VLM

My dream VLM has the following properties:

- Text and images enrich each other's representations instead of competing for resources
- Image-understanding is trained actively and end-to-end, embedded into surrounding context
- Image-understanding and -generation use as much of a common backend as possible
- Every trick that boosts intelligence is used, even if it is expensive

I believe that this is best achieved by using DeepSeek's [Janus](https://arxiv.org/abs/2410.13848v1) / [Janus-Pro](https://arxiv.org/abs/2501.17811) model with the following modifications:

- Merge modalities later at the input
- Split them earlier at the output
- Make the attention mask bidirectional for image-understanding
- Actively train image-understanding by de-noising learned mask-tokens
- Multi-scale prediction on masked image-tokens
- Image-generation using the image-understanding backend, the same mask-tokens, and a diffusion model

I need to stress again that these modifications are tuned for giving the *smartest* VLM, not the *fastest* one. There may be tradeoffs to consider, but I want somebody to go all in on intelligence. Optimizations are for the future.

This is the starting point (image from the DeepSeek Janus paper):

![Janus Architecture](images/JanusArchitecture.png)

> The [Janus Architecture](https://arxiv.org/abs/2410.13848v1).

## Split the heads earlier, merge later

To summarize [my previous article](https://snimu.github.io/2025/03/30/multi-layer-language-heads.html), the first transformer block (attention + MLP) approximately transforms the explicitly text-based representations into abstract ones, and the last layer transforms the abstract representation back into text-space. This is corroborated [here](https://x.com/jonasgeiping/status/1906667091324793307) by Jonas Geiping, co-author of [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://www.arxiv.org/abs/2502.05171) about looping a recurrent block.

The obvious consequence of this is that modalities, which look very different when instantiated into text/images/..., will try to push the model towards producing a representation most fit for producing exactly that modality, as opposed to others. This leads to competition for resources inside the model. Putting a transformer block per modality between the model core and the decoders will reduce that competition, and projecting into an abstract space should have a similar effect.

So the first modification that I would do to Janus is to merge inputs from different modalities later, and split their outputs earlier, like this:

![Janus Architecture with split and merge](images/abstract-space.png)

> The [Janus Architecture](https://arxiv.org/abs/2410.13848v1), adapted to have a common backend between modalities only in abstract space.

### Why attention + MLP instead of a simple linear layer?

I want to be very clear that this is pure intuition, so take it with a massive heap of salt.

Let's first look at the text-encoder only. Why does it only work in abstract space after the first transformer block? My answer is that abstract thought emerges from the connections between words/tokens, not the individual tokens themselves. Only by mixing the tokens along the sequence dimension via attention (or an SSM or whatever), and then along the model dimension via the MLP, can we get the abstract thoughts (I would say that the MLP does the actual conversion into abstract space, and attention is just there to steer it into abstract space for the right combination of tokens, but that's not super relevant and I'm also not very confident about it).

For images, it's similarly very difficult to move into abstract space without seeing the image as a whole, which only happens when the patches are mixed.

Therefore, I think *at least* one transformer block per modality at the input, and another one at the output, is needed to properly separate the specific quirks of each modality from the shared semantic understanding used and refined by the shared backend.

## Bidirectional image-understanding

My next modification is to make the attention mask bidirectional within the limits of a single image.

Yes, [PaliGemma](https://arxiv.org/abs/2407.07726) and [Gemma](https://huggingface.co/blog/gemma3) use it, which is a good reason to do it already; but I have more reasons than appeal to authority.

A causal attention mask for image inputs that are subsequently not used for anything makes no sense to me. Images aren't inherently autoregressive; they have no time-component shared by every image. That would be a video! So just make the mask bidirectional where the image is, and causal everywhere else. This way, every part of the image can see every other part of the image, which seems algorithmically important; and the attention mechanism at the image-tokens is used to full capacity, instead of half (as would be the case with a causal mask).

Those are the theoretical advantages of bidirectional masks, but the thing that made me *feel* those advantages was actually xAI's Aurora model, specifically this image:

![Arnold by Aurora](images/ArnoldByAurora.JPG)

> From [this post](https://x.com/fofrai/status/1867629384099934557?s=46) by [@fofrAI](https://x.com/fofrai)

Here is my speculation on why this image looks so weird, from [this post](https://x.com/omouamoua/status/1867926507391537260?s=46) by [@omouamoua](https://x.com/omouamoua) (that's me):

```txt
Regarding the weird hands:

I have a hypothesis that generating images autoregressively left-to-right, then top-to-bottom — as I think Aurora does — will lead to errors mostly at the bottom of the image.

Of course, the obtuse way to explain this is by saying that autoregressive errors accumulate.

But in the case of images, there is a more concrete explanation: the model generates plausible patches for the first row it creates. Here, that’s two hairlines from people at different distances to the camera; makes sense. But as the rows are created with plausible looking patches, one by one, a conflict emerges: both Arnolds intersect, and you need to resolve this in a plausible manner when you have never planned for it. That’s a high stakes situation where a single error can derail the whole image.

This is made worse by the tendency of undertrained models to make the highest-probability token pretty repetitive, and not as dependent on previous tokens as would be ideal. This is likely why there are two Arnolds in the image, and also why both must have a hand in the foreground, forcing the model to resolve this by screwing it up.

Essentially, a next-token predictor never explicitly learns to care about the composition of the entire output.

Multi-token prediction would fix this.
```

A bidirectional mask in an image *is* multi-token prediction, at least in a way. And yes, this is about image-generation, but 1) I want those advantages for image-generation too (see later how it's connected to image-understanding), and 2) I'm convinced that similar considerations apply to image-understanding as well.

Here's what this would look like:

![Bidirectional image-understanding](images/imun-bidir.png)

> Bidirectional image-understanding. On the left are the inputs, on the right the targets, and in the middle the transformer, represented by its attention mask. Image-tokens are shown as images for visualization, but are in practice flattened embeddings. The four-armed giraffe was generated by [Imagen-3](https://deepmind.google/technologies/imagen-3/)

### Efficiency and KV-cache

Just to stress: if we do this, we have to forward-pass the entire image at once. This is computationally expensive, but no more than passing a user-input with the same number of tokens as the image. And just like that user-input, we want to do it in parallel anyway, which means this costs us almost nothing compared to a causal mask.

And because we can just feed in the entire image at once, and the tokens *after* the image tokens are masked, we can use a kv-cache.

## Masked image-understanding

In Janus, DeepSeek doesn't decode the image-understanding-tokens, instead relying on subsequent text- and image-generation-tokens to provide the loss signal.

I see no reason not to actively train image-understanding by replacing some of the patches in the input sequence with learned mask tokens. Decode the hidden states into the same image-tokens but un-masked (and shifted by one to play nicely with the autoregressive nature of the rest of the model, as is done in [GPT or BERT: Why not both?](https://arxiv.org/abs/2410.24159) or, I think, [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961)), and we should get much better image-understanding. Masking image tokens is the task from [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), just without the encoder-decoder structure that saves compute. It should really help learn to understand images, especially since we work in abstract space.

![Masked image-understanding](images/imun-bidir-masked.png)

> Training bidirectional, masked image-understanding. On the left are the inputs, on the right the targets, and in the middle the transformer, represented by its attention mask. I show the images themselves being masked and produced at the output, but for image-understanding, what would be masked and produced would be adapted image tokens. After all, this is about image-*understanding*.

I would train a small percentage of the samples without masking, so that the model gets used to seeing the full image and using it for downstream tasks, and train the rest with between 0% and 100% of masked patches. It will become clear why I want to use up to 100% of the image-tokens when I get to image-generation.

### Should we even use SigLIP for this?

This is a question that I'm uncertain about. The idea behind it is: if we already actively train image understanding, do we not limit ourselves by using a frozen encoder that was trained by projecting into text-space, instead of training end-to-end?

Imagine the following architecture:

1. Image is split into patches
2. Patches are flattened
3. Flat patches are transformed into model dimension D via a Fully Connected (FC) layer
4. The result is projected into the abstract input space of the VLM by a transformer block (attention + MLP)

If this is trained end-to-end, with text and other images in context, then:

- There will inherently be a CLIP/SigLIP-like effect, because as long as the text is useful for predicting image-patches and images are useful for predicting text, their embeddings will be aligned automatically to be maximally useful to each other
- In [Scaling Language-Free Visual Representation Learning](http://arxiv.org/abs/2504.01017), the authors show that scaling self-supervised learning (SSL) methods in large transformers works better than CLIP. So why not save the parameters and just train end-to-end?

*Sidenote:* The authors of that paper use methods like DINOv2, not mask-denoising, but I expect that the latter would also work, and we will see why it is useful when we come to image-generation.

Some of you may fear representation collapse if we do this: the easiest way for the image-encoder to always be able to un-mask the image-tokens is to always produce 0 at its output. But as long as the images are useful for understanding the text coming after them, there is another loss signal working against representation collapse. This makes me believe that representation collapse is not a problem in such a VLM.

I don't know for sure if this will work, but I suspect so. From hereon out, I will assume that this is the case, and refer to the image-inputs as image-tokens.

## Multi-resolution, multi-scale image-understanding

Image understanding can be taken further by going multi-resolution / multi-scale:

![Bidirectional, denoising, multi-scale image understanding](images/imun-bidir-masked-multiscale.png)

> Multi-scale, masked-token, bidirectional image-understanding. Inputs on the left, targets on the right, transformer in the middle. The images are shown as images, and the mask tokens as red patches on top of them, but both are, in reality, flattened image-tokens.

Multi-resolution prediction is what is done in [Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness](https://arxiv.org/abs/2408.05446), where Stanislav Fort and Balaji Lakshminarayanan stack multiple resolutions of the same image along the channel dimension of a CNN, which significantly improves adversarial robustness of the models for basically no extra cost. The same advantage can be used in a transformer, by simply downsampling and masking the image. In transformer, this would of course incur additional costs, but I believe that it would be worth it. And since we can use fewer tokens for the downsampled images, the extra costs are not exorbitant.

*So what's a practical example where multi-scale image understanding would be useful?*

Anything where looking at the image at multiple resolutions is useful. For example, there was this brief trend of creating images that look like a normal photograph, but when you zoom out, the elements of the picture show a text, like this:

![OBEY](images/OBEY.png)

> If you look closely at the image, you will see a bunch of people walking around, but if you squint or look from far away&mdash;in other words, if you reduce the resolution of the image&mdash;the peoples' clothes spell out "OBEY". I don't know where the image originates from, so I unfortunately can't give a source.

Seeing this image in its original resolution *and* a highly downsampled version will allow a model to see both aspects of it at once.

Another example for a task that would benefit from a multi-resolution/multi-scale approach is recognizing people/objects in noisy images, such as this one:

![Noisy image recognition](images/NoisyImageRecognition.jpg)

> From [this post](https://x.com/torchcompiled/status/1909878529321222619?s=46) by [@torchcompiled](https://x.com/torchcompiled).

It's pretty easy for humans to recognize this image, but difficult for current VLMs. The difference, I believe, is that humans use multi-resolution vision.

These examples are somewhat toy-like, but they point to an important capability: reliability in image recognition, and an ability to handle complex, previously unseen contexts.

## Add a diffusion model

Now, all we have to do to to generate images, too, is to add a diffusion model and separate it from the common backend with its own transformer block. We use the activations of the transformer to guide the diffusion model. For every single image that we see, consisting of between 25% and 75% mask-tokens, we apply both the image-understanding head and the image-generation head (a.k.a. a transformer block + diffusion model). This is illustrated below:

![Multi-decode](images/imgen-multi-decode.png)

> Decoding the common hidden states into image-tokens and an actual image via two independent heads, separated by their own transformer blocks.

I will get to the details of training and inferencing this setup, but first, some general advantages that we can already determine:

- Multi-scale prediction seems to work very well. It allows the model to first generate rough outlines of an image at low resolution, then refine them step-by-step.
- We are using multi-scale generation, so a diffusion model, which is naively best at the very local and high-frequency components of an image, can still get the low-frequency, rough outlines right.
- This method of generation allows us to use a bidirectional mask, which should be superior to a causal mask not just in understanding the image at the previous scale, but also at generating good representations at the current scale. It also means that we can generate the hidden states for the entire image at once (per resolution), instead of token-by-token, and then just use a cheap diffusion model for a few steps on top.
- We get all the advantages of diffusion models:
  - Image diversity / easy re-generation with a different noise seed from cached hidden states
  - Great aesthetics
  - Easy adaptability of those aesthetics via cheap LoRAs to the diffusion model alone
  - Multi-step generation for a good efficiency-to-quality trade-off

There are a few strange things about this.

*Why do we do image-generation from only 25%-75% mask-tokens?*

During inference, we won't, but I will get to that shortly.

During training, it offers multiple advantages:

- Because we use the image-encoder for both image-understanding and -generation, we don't need any text-labels. After all, 25%-75% of the image are still visible, so the hidden states can act as meaningful conditioning. Of course, we *can* still use text-labels, in interleaved text and image data.
- We enable inpainting. Just draw mask-tokens into your image, feed it through the image-encoder and common backbone, and apply the diffusion model.
- The image-understanding head is trained in abstract space. To avoid representation collapse, predicting into pixel-space with another head will help, so we don't need as much interleaved text and image data to achieve the same, for example allowing us to train the modalities independently at first if we have a lot of text-only or image-only data.
- We can always train both image-understanding and image-generation for every image, which does two things:
  1. It gives a ton of training signal
  2. It aligns the two, so that they can work well together
- And just to repeat myself, the transformer blocks separating the two should mean that they don't compete for resources, instead training the common backend to produce highly meaningful representations.

*What about inference?*

We are currently training with 25-75% mask-tokens, but actual image-understanding should happen with 0% mask-tokens, and image-generation with 100% mask-tokens. How do we do that?

The simple answer is: post-training.

For image-generation, train a bit with 100% mask-tokens at the input. After every image-scale, swap the mask-tokens out for the actual image, then generate the next scale. By using 100% mask-tokens at the input, we don't need any pre-existing image. And since we can simply feed in all the mask tokens for a certain resolution in parallel, this is a two-step process per scale: 1) generate image from mask-tokens; 2) swap out mask tokens for actual image and generate. However, step 2 can be done at the same time as the next scale is being generated, which saves even more steps, so this is closer to typical input-tokens than output-tokens in text.

Here's what that would look like:

![Image generation](images/imgen-inference.png)

> Multi-scale, masked image generation at inference time. To keep the image readable, it does *not* show the diffusion model after the transformer that generates the actual image, nor the transformer block that separates the common backend from the image-understanding- and text-decoders. You will notice that there are some additional tokens before, after, and between the different image-scales; these are needed to line up the inputs, outputs, and attention mask.

You may notice that this is basically [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905), with a few changes on top.

As for image-understanding, it's even simpler: post-train on a bunch of interleaved text and image data with 0% mask-tokens over the images, then use the model just like that during inference.

However, there is another way to achieve the same.

## Add the un-masked image

If we simply add the un-masked image after the masked image at every scale, but only decode with the image-understanding and -generation heads on the masked images, at every resolution, we get an even better model, and can get rid of the crutch of replacing mask-tokens with the actual image in image-generation.

First off, here's what that would look like:

![Image generation: training](images/imgen-training.png)

> Training a multi-scale, bidirectional-per-scale, masked-token image-generation model. The diffusion model and image-understanding head are not shown here, but their training can of course be parallelized.

There is a single disadvantage to this approach: cost. The number of tokens per image is doubled. Of course, as I've written above, these tokens should be treated more like input-tokens than output-tokens, even in image-generation, so at reasonable sequence lengths, this shouldn't be a huge issue. For inputs containing many images, though, these extra tokens will be felt.

As for advantages:

- During inference, you can simply append the generated image to the mask-tokens, so the image-generation model inherently sees its own creation in its un-masked form at every scale, which allows for three things:
  1. The image can be used to guide the generation of the next higher resolution
  2. The un-masked image is immediately visible to the model for downstream text or image tasks
  3. The model can see the generated image in full resolution right after having generated it at every scale, and (given the right tools) decide that there is something wrong with the image. Then, you could simply re-generate the image by giving the diffusion model a different noise-input, guided by the same hidden states. How to make that decision is a bit of a difficult question, but I would guess that it would be possible in some way. This would be cheap because only the diffusion model would have to be run again. To be clear, I don't mean generating *all* resolutions and *then* re-generating; I mean something like evalutating the difference between the activations for the mask-tokens and the actual image at those positions, and re-generating if the difference is too large.
- In fact, I want to stress this: for the purpose of downstream tasks, image-generation is exactly equivalent to image-understanding.
- As for image-understanding, we can now train that capability with a pretty heavily masked image, and still have the CLIP-like effect of having the full-resolution image in context for downstream text generation.
- In both cases, this setup allows the model to first spell out what it thinks should be under the mask tokens (be there 25% or 100% mask tokens, or anything between), and then see what is actually under them.
  - In the case of image understanding, this allows the model to use surprisal as a strategy for analyzing the image. And yes, I'd use that during inference: mask the image 25-75% (the exact number is an emprical question) at the position where it is masked during training, and leave it un-masked where it is un-masked during training.
  - In the case of image generation, this of course does the same, but with the right training data, it might also allow for something else: the model correcting itself during generation (beyond just re-generating the diffusion output).
- There is a small chance that the mask-tokens act a bit like registers, see [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).

## Conclusion

While I have no guarantees about any of these modifications, I would love to see someone with a budget and the skills to try them out:

- Merge later and split earlier for thinking in abstract space
- Mask tokens and a bidirectional attention mask for image-understanding
- Multi-scale prediction for image-understanding
- A diffusion model on top of the image-understanding-backend for image-generation

![Full pipeline](images/full-pipeline.png)

> Proposed modifications to the Janus architecture, rough sketch.

## Citation

```bibtex
@misc{snimu2025perfectvlm,
    title={The perfect VLM},
    author={Sebastian M\"uller},
    year={2025},
    month={04},
    url={https://snimu.github.io/2025/04/07/my-dream-vlm.html}
}
```
