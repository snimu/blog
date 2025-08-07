# Multi-Frame Prediction for World Models

Video models (or world models if they have an additional action input) often struggle with scene consistency over long periods of time, and with natural-looking movements. In this post, I speculate about predicting the next frame in full resolution.

## The issue

Video models often produce very beautiful looking frames, but the movements being depicted are anywhere from creepily to ridiculously wrong. This is quickly changing (thank you Google), but I don't know how it's done. The likely answer is more and better data, but I do wonder if multi-frame prediction is a way to make this more data-efficient.

I see the issue stemming from two sources:

1. The short-sightedness that next-frame prediction inherently introduces. Subtle errors will accumulate, and models aren't taught to correct course because they don't care about the consistency between frame `n` and frame `n+32`, just between frames `n` and `n+1`.
2. The high resolution of the predicted next frame. Models are often biased toward high-frequency components of data&mdash;the local details&mdash;and those are typically not the main component of movement. A video with significant random noise can still display movement just fine, but a model predicting the next frame in high resolution would be punished harshly for wrongly predicting the noise, without being punished for missing the long-range dependencies that make up the movement. If the noise isn't fully random but somewhat predictable (for example, high-frequency features in real images), the model would therefore not learn to predict the low-frequency movement across frames well, but mostly the high-frequency noise.

## Solutions

The solution I see for these issues is twofold, with two optional extensions:

*First, predict multiple frames ahead*; and not just frames `n+1` through `n+4` (for example), but (for example) frames `n+1`, `n+2`, `n+4`, and `n+8`. This forces the model to represent a scene internally in a way that enables it to predict pretty far ahead, capturing long-range movements. Of course, it's possible to be more extreme and predict `n+1`, `n+16`, `n+256`, and `n+4096` frames ahead (or whatever you like), but the exact numbers don't matter for the purpose of this post; they are best left to experimental validation.

*Secondly, make all predictions except for the first one low-resolution*. This would fight problem number 2, because downsampling removes noise by averaging it out, and for the purpose of predicting movements, details are (mostly) noise. It's also cheaper than predicting at full resolution. Yes, we could predict everything at low resolution and throw an upsampler on top of it, but that's not an end-to-end solution so I'm not a fan (specifically, I believe that there is a lot of valuable signal in the high-frequency components of an image that the world-model could learn from, and externalizing the prediction of those would probably lead to a worse world model and less consistency of the details between frames). So the first predicted frame, being the most important one, should be predicted at full resolution, and all others at lower resolutions.

*Extension 1&mdash;Auxiliary Loss*: To really make the model care about the consistency of movement between all the frames it predicts, we could introduce an auxiliary loss that compares the consistency between the low-resolution prediction `8` frames ahead at position `n` and the downsampled prediction `1` frame ahead at position `n+7`. This would force the model to not only care about the similarity of its predictions to the ground truth, but also about self-consistency, by introducing a dependency between the prediction one frame ahead and the one multiple frames ahead.

*Extension 2&mdash;Multi-Resolution for every frame*: Instead of predicting the full resolution a single frame ahead, and the subsequent frames at lower resolution, predict them all at multiple resolutions like in [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905v2). This would defeat the disadvantages of predicting at high resolution, while keeping its advantages; while retaining the benefits of multi-frame prediction of far-out frames (like 16 frames ahead or more). That would be extremely expensive, though, so one might predict all resolutions for frame `n+1`, all but the most detailed one for frame `n+2`, all but the two most detailed ones for frame `n+4`, and so on. That's still pretty expensive, but much less so.

## Summary

I believe that multi-frame prediction for video/world models is highly beneficial. These predictions shouldn't just be directly adjacent frames; instead they should leave growing gaps between the multiple predictions per frame. I also believe that all but the first prediction ahead should target low resolution representations of the target frames, to focus on movement and long-term planning instead of visual fidelity (the latter of which is already covered in the first prediction ahead). Finally, an auxiliary loss and per-frame-multi-resolution may or may not provide further help.
