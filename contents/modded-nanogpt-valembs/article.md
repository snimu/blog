# modded-nanogpt: Embeddings Extravaganza 1 (value embeddings)

Adding more value-embeddings to [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) led me to an (as of yet unofficial) modded-nanogpt medium record.

In this article, I present these results, and a lot more ablations of experiments that didn't work out. Here's a table of contents:

- [The record](#the-record)
- [Adding value embeddings](#adding-value-embeddings)
- [Removing value embeddings](#removing-value-embeddings)
- [Shifting value embeddings](#shifting-value-embeddings)
- [Sharing value embeddings differently](#sharing-value-embeddings-differently)

You can find the reproducible code [at this link](https://github.com/snimu/modded-nanogpt-experiments/tree/main/experiments/00003-value-embeddings).

## The record

The best setting I've found is adding two more value embeddings, in the same tied manner as before. The baseline has the following value embeddings:

- One applied to layers 0 and 13
- One applied to layers 1 and 14
- One applied to layers 2 and 15

I have added two more value embeddings in the following way:

- One applied to layers 0 and 11
- One applied to layers 1 and 12
- One applied to layers 2 and 13
- One applied to layers 3 and 14
- One applied to layers 4 and 15

I re-ran the baseline, because I slightly changed the compile flags and used a newer version of PyTorch. Here is a plot of the value losses over time, averaged over 28 runs for the baseline and 37 runs for the record:

![New record: validation loss over time](images/25-26-record-time.png)

With the two additional value embeddings, the model reaches the target loss in ~35 fewer seconds.

> A note on the averaging: I very simply averaged the loss for each training step, and independently averaged the time taken at each training step, and then plotted the loss over the training time. That's not 100% mathematically correct because I'm averaging losses that happended after different amounts of time, but the averaging of the times should mostly make up for that, so the results are still valid (especially considering the large margin with which the record is set).

The record plot ends earlier because I ran it for fewer steps, 5820 instead of 5890, so all the benefit comes from improved per-step learnability. The per-step time is slightly increased by the new embeddings, as expected. Here is a plot of the validation losses over the steps:

![New record: validation loss over steps](images/25-26-record-step.png)

I clearly could have stopped earlier, but I wanted the record to be rock-solid, because it is a requirement for an official record to show that the final validation loss of the runs is below 2.92 with a very high probability.

Here are the final validation losses over all 37 runs that I've performed:

```python
[2.919612, 2.919458, 2.918941, 2.917664, 2.91856, 2.919706, 2.919218, 2.918082, 2.919345, 2.920486, 2.919293, 2.917286, 2.921162, 2.919861, 2.917587, 2.919488, 2.919955, 2.919172, 2.919245, 2.918839, 
2.918381, 2.919301, 2.917944, 2.919178, 2.918395, 2.920141, 2.918754, 2.918432, 2.919958, 2.91978, 2.919916, 2.919711, 2.918025, 2.919342, 2.920571, 2.917387, 2.919093]
```

And here are some simple statistics about those losses:

- Mean:   2.919 ± 0.001
- Median: 2.919
- Min:    2.917
- Max:    2.921

The probability of the average final loss being below 2.92 is ~99.99995%.

## Adding value embeddings

What happens if we add more value embeddings? In the following plot, we see the validation losses over training of runs with a total of three (baseline), four (see above), five, six, seven, and eight value embeddings, each shared like in the baseline (so applying the n value embeddings to the first n layers, and to the last n layers again in the same order). That means that the one with the most value embeddings applies one value embedding to each layer in the model (except layer 7 which is attention-free). These plots are for only a single run each.

![13, 15, 16, 17, 18, 19](images/13-15-16-17-18-19-time-1200-1500.png)

Here is the order in which the runs cross the 2.92 loss-barrier, which is the target of modded-nanogpt medium:

1. 2 additional value embeddings (1410 sec ~= 23.5 min)
2. 1 additional value embeddings (1422 sec ~= 23.7 min)
3. 5 additional value embeddings (1425 sec ~= 23.75 min)
4. 3 additional value embeddings (1428 sec ~= 23.8 min)
5. 0 additional value embeddings (1436 sec ~= 23.9 min) (baseline)
6. 4 additional value embeddings (1444 sec ~= 24.0 min)

However, these times will of course vary over different training runs, and occur only in the last ~50s. In the time before that, which tells us more about what setting is consistently the best, this is the order from best to worst:

1. 2 additional value embeddings
2. 1 additional value embeddings
3. 0 additional value embeddings (baseline)
4. 3 additional value embeddings
5. 5 additional value embeddings
6. 4 additional value embeddings

Let's see if adding more and more value embeddings consistently improves the amount that is learned per batch, by plotting the loss over the training steps:

![13, 15, 16, 17, 18, 19](images/13-15-16-17-18-19-step-5000-6000.png)

This seems almost monotonous, except for adding four additional value embeddings, which is the worst. Here is the order in which they cross the threshold:

1. 5 additional value embeddings
2. 3 additional value embeddings
3. 2 additional value embeddings
4. 1 additional value embeddings
5. 0 additional value embeddings (baseline)
6. 4 additional value embeddings

I would interpret this as "more value embeddings lead to more learning per step", but the training run when I added 4 additional value embeddings was an outlier. I assume it would fit in nicely with the others in the trend if run multiple times.

I assume that the 4 additional value embeddings slot in nicely in this timed order (from best to worst): 2-1-0-3-4-5. So for one and two additional value embeddings, the additional loss reduction per step dominates over the additional per-step time, while above two additional value embeddings, the effect reverses.

## Removing value embeddings

If adding value embeddings helps performance per time-step, then removing them should hurt them. Nevertheless, my [investigations on different learned scalars in modded-nanogpt](https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html) led me to wrongly suspect that it should be possible to remove value-embeddings, especially in the early layers where their effects are strongly suppressed, which would of course speed up the runtime because it would avoid some computation (though only a small amount).

So, for the sake of completeness, here are the results of those experiments as well, starting with the removal of single value embeddings:

![0-5, 13](images/0-1-2-3-4-5-13-time-1250-1500.png)

None of them is as good as the baseline. However, the ordering of results is interesting. Keepingin mind that random noise is a big factor here, it seems the order of removing value embeddings from worst to best is 2-13-14-15-0-1. Well that's confusing.

Removing layer 2 is the worst by far? But the other two early layers are of almost no importance? That's strange, and it makes me wonder what would happen if we shifted the value embeddings to slightly later layers; so instead of layer `[0, 1, 2, 13, 14, 15]` we'd have layers `[1, 2, 3, 13, 14, 15]` or `[2, 3, 4, 13, 14, 15]`.

Before that, let's look at what happens if we remove multiple layers at once though. First, removing layers 0 and 1, and layers 0, 1, and 2. To be clear, no value embedding is removed for this, because they are shared with the value embeddings at layers 13, 14, and 15, and those remain; they are simply not applied to the marked layers. For comparison, I'll throw in the baseline and the best performing run from before (so only layer 1 removed):

![1, 6, 7, 13](images/1-6-7-13-time-1200-1500.png)

The two conclusions I can draw are that (1) the early value embeddings are definitely valuable and (2) the more of them are removed, the worse the result.

How about removing full value embeddings? So removing the shared value embedding from layers 0 and 13, or 1 and 14, or 2 and 15; this time compared to removing layers 0 and 1 (because that's also not adding value embeddings at two layers, but it keeps all the parameters):

![6, 8, 9, 10, 13](images/6-8-9-10-13-time-1200-1500.png)

Removing a full value embedding seems to be worse than not sharing its weights, though it's difficult to properly compare because it's applied to different layers which, as we have seen, has a big effect. I haven't run any experiments looking at, for example, removing the value embeddings from layers 0 and 14, or 1 and 13, etc., but those would be better comparisons. There is no real trend here where we could say "removing later layers is consistently worse" or something similar.

## Shifting value embeddings

As discussed above, it might be interesting to shift the value embeddings from the early layers to slightly later layers. Let's see this on two examples, where we use the value embeddings on layers 1, 2, and 3 or 2, 3, and 4 instead of 0, 1, and 2. Since I expect any time-differences to be due to random chance, I will just plot them over the steps:

![13, 20, 21](images/13-20-21-step-5000-6000.png)

The shifted value embeddings seem to lead to worse results than the non-shifted ones, and moreso the more they are shifted.

## Sharing value embeddings differently

I did wonder if the structure of sharing value embeddings across many layers (so layer 0 and 13 etc.) is optimal, or if we can share them differently. So I tried limiting myself to two value embeddings, which were shared in the following ways: (1) layers 0&1 and layers 14&15 share a value embedding each, and (2) layer 0-2 and layers 13-15 share a value embedding each. This removes the parameters of a whole value embedding, so I'll keep the run where I removed the value embeddings from layers 0 and 13 (and thus removed a full embedding layer):

![8, 11, 12, 13](images/8-11-12-13-time-1200-1500.png)

When we remove a full embedding layer, it is better to keep the interleaved structure where the value embeddings are shared between early and late layers, not between adjacent layers. Additionally, it is worse to share the same value embedding between three layers than it is to share it between two layers, even though the toal number of parameters stays constant.

Another way of sharing the value embeddings differently is to let invert the application of the value embeddings in the early layers. I performed this experiment with the full five value embeddings, and over 17 runs.

Before, the value embeddings are tied like so:

- Embedding 1 is applied to layers 0 and 11
- Embedding 2 is applied to layers 1 and 12
- Embedding 3 is applied to layers 2 and 13
- Embedding 4 is applied to layers 3 and 14
- Embedding 5 is applied to layers 4 and 15

I thought it might be better to invert this like so:

- Embedding 1 is applied to layers 0 and 15
- Embedding 2 is applied to layers 1 and 14
- Embedding 3 is applied to layers 2 and 13
- Embedding 4 is applied to layers 3 and 12
- Embedding 5 is applied to layers 4 and 11

This did not work though: over 17 runs, the mean final validation loss is 2.9196, compared to the 2.9194 without this change. This is a tiny, tiny bit worse, which means to me that the edit either changes nothing or makes things worse (though it's hard to tell with such small differences).
