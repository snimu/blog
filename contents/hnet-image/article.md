# H-Net VLMs: some loose thoughts

H-Net (LINK) is a brilliant paper that ...

It is currently defined for text only. In this article, I will speculate how one could extend it to images, video, and world models (so video + action inputs).

## H-Net: summary

I will give a very short summary of the H-Net architecture here, but I strongly recommend that you read the actual paper (LINK).

...

## Easy wins

The simplest way to do an H-Net VLM is to smash the text- and image-bytes into the same stream and run a normal H-Net over them. Let's get some easy improvements to this paradigm out of the way.

First off, bytes of images should have different embeddings than bytes from text. TODO

Secondly, (different learned vector when extending img token dim than when extending txt tok dim)

## More speculative improvements

Now, I want to discuss more speculative, complex improvements to the pipeline.

### Separating text and images

Text is a far more compressed modality than images, to the point where raw text-bytes express significantly more information than raw image-bytes. Therefore, simply putting both together at the same input can lead to problems. The solution is to first compress the images into tokens of approximately the same information density, or "meaning-compression", as the text-bytes, and only then merge them.

What I imagine doing is running an H-Net encoder and chunker over all the images independendly in a batched manner, then flatten the images and put them in the right position in the image-text-sequence, and only then running the normal H-Net pipeline.

TODO: IMAGE

And at the output, the inverse would have to be done: after the last decoder for the text-bytes is run, I would extract the bytes belonging to each input image, and expand them in a batched manner with their own decoder and dechunker layer.

TODO: IMAGE

In fact, at the output, I would go a step further and separate the modalities even more strongly. My intuition for doing so is that the output latent will be decoded directly into the actual output (text, and image, ...), so it should be optimized to make it as easy as possible to decode it into *the specific output modality*. If the same latent is decoded into multiple modalities which are significantly different, then it has to fulfill two very different jobs and cannto be as specialized anymore. That still works, because the latent can learn to be the supreposition of two discrete latents which are distinguished by the prediction head of each modality, but it's suboptimal.

To prevent this issue, I would instead only share the same layers between the modalities up to some more abstract latent, deeper inside the model, and then run individual decoder layers on each. Concretely, I'd do the following

- Run the combined text- and image-sequence through three out of the four decoder layers (they use four in the paper so I'll stick with it)
- Run the combined text- and image-sequence through a fourth decoder layer (keep the images for maximum context to the text), then throw away the images, decode the text-predictions with a text-head, and apply a loss only for the text-predictions
- Run the combined text- and image-sequence through a different fourth decoder layer (keep the text for maxmium context to the images), then throw away the text, decode the images by de-chunking them in a batched manner and running another decoder over them, and apply a loss only for the image-predictions

TODO: IMAGE

This has multiple advantages:

- The predictions are made from a shared abstract representation, which will be strengthened by the combined gradients from both modalities
- At the same time, the output latents for text and images can be produced from the shared latent from individual weights, allowing a specialization to each modality
- And yet, both still have the full context in both modality-branches of the decoder

### Bidirectional attention for images

TODO

- flattened image + causal pred = strong bias which is unnatural
- bidirectional attention per image is better
- solution:
  - In the batched encoding and chunking layer, use bidirectional attention instead of Mamba (you can have like two layers of attn then two layers of Mamba for its bias toward compressing representations which is good for chunking, but starting with bidir attn for understanding the imag in a more natural way, which should help the Mamba layers and the chunking and the representation given to the next step)
    - preserves temporal causality
    - without treating 2D space the same way as 1D time, which would be dumb
  - At the output, just do the inverse
    - for training, this obviously just works
      - though it makes image prediction trivial
      - solution: heavy masking at input
    - for inference, it's completely fine if we simply want to understand an image
    - but that violates causality if we want to generate one!
      - solution: add a VAE loss to the image-only parts of the model
      - this forces it to have the same representation at the output of the image-only-encoder and the input of the image-only-decoder
      - which would enable us to simply loop the latents in the autoregressive part of the model, then decode all the generated latents in parallel with the bidirectional attention decoder
      - just run the full model including the encoder once on the generated image after it has been generated to update the align the model's internal representations with what it has actually generated
      - question: would this also encourage the text part to share a representation? after all, the use the same backend. the splitting of modalities at the output might mean that no, it doesn't, but if it does, it would immediately enable us to do latent looping there as well (need to specifically train it of course)
- H-Net provides a unique way to add autdio-information to the model:
  - Deeper stages of the model have a higher model dimension than ones closer to the in- and output
  - For reducing the dimensionality, this switch is done by just cutting away the excess part of the vector (I have ideas for what to do with those parts, but I'll get to them in another article)
  - For increasing the dimensionality, the missing part of the vector is added by concatenating a learned vector of the dimensionality that is missing
  - This vector could encode information about sound in the video!
  - We can get the vector by running another, smaller H-Net in parallel that just encodes the audio
    - appending a single vector three times per forward pass requires very little communication, so the second model could be run in parallel very easily
  - Audio and video are always synched, so adding one audio vector per video frame makes sense
  - If the reduction / increase in dimensionality is small, the audio model can be much smaller than the video model (it can also be smaller otherwise by just reducing its dimensionality)
  - The only issue I can see with this is that we'd either have to chunk the audio-model outputs the same way as the video model outputs, or interpolate between the chunks in an appropriate way, which would add a lot of complexity

### Use a diffusion model

TODO

### Extending to video

TODO

- bidir etc. still great
- but is there a way to merge the same patch over time?
  1. patch images
  2. merge each patch with the same patch from the other in batched manner
  3. bring the patches of each frame back into image shape, then bidirectionally merge over space
      - how would you handle patches that are now merged across multiple images?
      - probably by smoothing / interpolation
      - but have to think more on this
  4. Bring the patches into the normal autoregressive order and apply the normal H-Net
  - This is very complicated, so not sure if it's worth it

### World models

- Are just video models with additional action input which gets added to the frame representations
- But you can add the action information to the model just like the audio information above
