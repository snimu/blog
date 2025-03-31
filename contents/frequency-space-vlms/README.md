# Idea: Frequency space VLMs

(status: wild speculation)

**The problem**

LLMs benefit greatly from task examples---the more, the better (mostly).

However, images are very expensive because they take many tokens to encode. Using something like [DSPy](https://github.com/stanfordnlp/dspy) to optimize a prompt with images can quickly ruin you financially if you want to use several examples. Additionally, the image tokens will fill up your context window much faster than the text tokens, which is bad for inference.

It might make sense to optimize which patches to use for the example. After all, not every part of an image is relevant for a given task. However, this seems difficult to implement.

An alternative is to downsample the image to a smaller size. However, some tasks might need the fine resolution of a large image, which you would ruin by downsampling.

**The solution**

Here is what I propose:

- Train the VLM in frequency space, i.e. on the Fourier transform of the image.
- Fine details of a high-resolution image are the high-frequency components of the Fourier transform, and low-resolution details are the low-frequency components (speaking roughly).
- Now, you can downsample the image to a smaller size, but in frequency space, where you can downsample by cutting out frequency bands instead of averaging over pixels.
- If a model is trained properly, it might be able to handle computing an image stiched together from different frequency bands.
- Then, it should be possible to optimize the frequency bands to use for each example, which is equivalent to downsampling but without losing the high-frequency details.
- During inference, you can use the full image, with all frequency bands, and the model (if trained properly) will be able to handle it.

**Will I test this?**

No.

**Citation**

To cite this article, please use the following BibTeX entry (or adapt it similarly):

```latex
@misc{mueller2024frequencyspacevlms,
    title={Frequency space VLMs},
    author={Sebastian M\"uller},
    year={2024},
    month={sep},
    url={https://github.com/snimu/blog/blob/main/contents/frequency-space-VLMs/README.md}
}

