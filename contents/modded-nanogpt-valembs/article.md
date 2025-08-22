# modded-nanogpt: Embeddings Extravaganza 2 (value embeddings)

TODO

## Adding value embeddings

Let's add another value embedding, so that now layers 0&12, 1&13, 2&14, and 3&15 each share one value-embedding. Here are the results plotted over time (cut to the later parts of training so that we can differentiate between the baseline and the new setting):

![13-15](images/13-15-time-1100-1500.png)

Adding another value embedding *obviously* immediately sets a new modded-nanogpt speedrunning record!

So what happens if we add more value embeddings? Let's add some, for a total of three (baseline), four (see above), five, six, seven, and eight value embeddings, each shared like in the baseline (so applying the n value embeddings to the first n layers, and to the last n layers again in the same order):

TODO

...

## Removing value embeddings

If adding value embeddings helps performance per time-step, then removing them should hurt them. Nevertheless, my investigations on different learned scalars in modded-nanogpt (LINK) led me to suspect that it should be possible to remove value-embeddings, especially in the early layers where their effects are strongly suppressed, which would of course speed up the runtime because it would avoid some computation (though only a small amount).

So, for the sake of completeness, here are the results of those experiments as well, starting with the removal of single value embeddings:

![0-5, 13](images/0-1-2-3-4-5-13-time-1250-1500.png)

None of them is as good as the baseline. However, the ordering of results is interesting. Keepingin mind that random noise is a big factor here, it seems the order of removing value embeddings from worst to best is 2-13-14-15-0-1. Well that's confusing.

Removing layer 2 is the worst by far? But the other two early layers are of almost no importance? That's strange, and it makes me wonder what would happen if we shifted the value embeddings to slightly later layers; so instead of layer `[0, 1, 2, 13, 14, 15]` we'd have layers `[1, 2, 3, 13, 14, 15]` or `[2, 3, 4, 13, 14, 15]`.

Before that, let's look at what happens if we remove multiple layers at once though. First, removing layers 0 and 1, and layers 0, 1, and 2. To be clear, no value embedding is removed for this, because they are shared with the value embeddings at layers 13, 14, and 15, and those remain. For comparison, I'll throw in the baseline and the best performing run from before (so only layer 1 removed):

![1, 6, 7, 13](images/1-6-7-13-time-1200-1500.png)

The two conclusions I can draw are that (1) the early value embeddings are definitely valuable and (2) the more of them are removed, the worse the result.

How about removing full value embeddings? So removing the shared value embedding from layers 0 and 13, or 1 and 14, or 2 and 15; this time compared to removing layers 0 and 1 (because that's also not adding value embeddings at two layers, but it keeps all the parameters):

![6, 8, 9, 10, 13](images/6-8-9-10-13-time-1200-1500.png)

Removing a full value embedding seems to be worse than not sharing its weights, though it's difficult to properly compare because it's applied to different layers which, as we have seen, has a big effect. I haven't run any experiments looking at, for example, removing the value embeddings from layers 0 and 14, or 1 and 13, etc., but those would be better comparisons.
