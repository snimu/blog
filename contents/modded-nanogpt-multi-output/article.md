# modded-nanogpt medium World Record: Reusing intermediate activations in the output latents

I have a new world record for the modded-nanogpt speedrunning medium track.

It involves the re-use of activations from earlier layers at the output latent via a learned weighted sum.

This article will describe that record, as well as multiple other experiments.

## The record

I will present the record in three steps: (1) reproducing the baseline results, (2) explaining the architectural change made to produce the record results, and (3) showing the record results.

### Baseline

The baseline comes from [PR#137](https://github.com/KellerJordan/modded-nanogpt/pull/137). I re-ran the code for 5 runs and got the following results:

- Mean final validation loss: 2.9191
- Mean time: 1393.16 seconds ~= 23.22 minutes
- T-test p-value: 0.01194797508928048

So after only 5 runs, we can be ~98.8% sure that the mean loss over an infinite number of runs would be below 2.92, our target loss.

### Record - Technique

I simply added the output of layer 11 (the 12'th layer) to the output of the final layer (layer 15, or the 16'th layer), in a weighted sum. This just means that right before applying the language head, I do this:

```python
skip_lambdas = self.scalars[-2:]
x = norm(x) * skip_lambdas[0] + norm(skip_connections[11]) * skip_lambdas[1]
```

Some details:

- `skip_connections` contain the output latents of each layer, at the corresponding position
- The `skip_lambdas` are learned scalar values; they are initialized to 1.0 for x and to 0.0 for the skip connection, and then actively optimized over the course of training

### Record - Results

Doing this allowed me to reduce the step count from 5590 to 5550, which lead to the following results.

These are the resulting final validation losses over 10 runs:

```python
[2.919485, 2.918384, 2.918878, 2.918476, 2.920099, 2.919609, 2.918705, 2.91872, 2.919772, 2.918594, 2.917798, 2.919295, 2.920676, 2.919743, 2.920052, 2.919843, 2.920081, 2.919675, 2.919486, 2.919177, 2.919529, 2.919678]
```

And here are simple stats about the final validation loss over these 10 runs:

- Mean: 2.9194
- Std: 0.00069
- P-value: 0.0001256

Now here are the corresponding run times:

```python
[1384.256, 1384.324, 1384.185, 1383.412, 1392.184, 1392.305, 1383.552, 1383.785, 1383.811, 1383.785, 1383.434, 1383.753, 1383.082, 1383.284, 1383.827, 1385.682, 1383.579, 1383.422, 1383.467, 1385.108, 1383.398, 1384.058]
```

And here are some simple stats about the times:

- Mean: 1384.6224 seconds ~= 23.08 minutes
- Std: 2.5382 seconds

This is a reduction in final run time of ~8.54 seconds.

### Lambdas

We don't just perform a sum between the outputs of layers 11 and 15, but a weighted sum; and those scalar weights are learned. So what values do they take?

TODO: add average over many runs; add plots of lambdas over the course of training

Here are the mean final lambdas over the course of 22 runs, rounded to 3 digits:

- Layer 15 (x-lambda): 0.802
- Layer 11 (skip-lambda): -0.279

This all but confirms a hypothesis by [Larry Dial](https://github.com/ClassicLarry) which he shared in a [comment](https://github.com/KellerJordan/modded-nanogpt/pull/138#issuecomment-3362739273) in the first PR I made about this record (which I closed to re-open a new one, because the first one was sloppy). His hypothesis is this (in my own words):

Every layer that is not the output layer has the job of providing context to the next layer so that it can do its job better. But each layer output is also present in the final output latents, due to the residual stream. Thus, it directly impacts the final prediction, and the layers all perform the dual jobs of providing context and making a prediction, which might not always be the same. Thus, each layer effectively performs the two tasks of prediction and information up-cycling, which might be in conflict with each other.

The final lambdas in these experiments are evidence for that hypothesis: the output of layer 11 is actively removed from the residual stream after layer 15, which should allow layer 11 to only focus on providing context to layer 12.

Let's see how the lambdas develop over the course of training, showing the lambdas for all runs, and their means:

![Lambdas over training steps](images/lambdas.png)

Two things become very clear:

1. The lambdas develop to the same final values very reliably
2. They develop in a very smooth fashion over the course of training

### Norms

Before my record, `x` was RMS normed right before applying the language head:

```python
...  # apply the layers
x = norm(x)
...  # apply the language head
```

I created an adapted version, in which I multiply `x` by a scalar value before decoding it:

```python
...  # apply the layers
x = norm(x) * l_x + norm(x_skip) * l_skip
...  # apply the language head
```

where `l_x` and `l_skip` are the abbreviated lambdas.

Would norming again after the sum help? Theoretically, the language head should be able to just learn to incorporate the constant factor from the lambdas and be fine, but sometimes learning dynamics are weird and don't work out like that. So I tried the following:

```python
...  # apply the layers
x = norm(norm(x) * l_x + norm(x_skip) * l_skip)
...  # apply the language head
```

But it made no difference at all, so it's fine to leave out this last norm. If anything, it made performance worse; but only very, very slightly, which means nothing when done for a single run, so I wouldn't take that too seriously.

## Adding more than one layer output

TODO: everything!

TODO: iff final two or three layer lambdas are positive and the rest are negative&mdash;a.k.a. the final few layers are actively doing prediction, while the previous ones only provide context&mdash;could we simply run them in parallel and then do a learned weighted sum over their outputs?

TODO: Are the magnitudes of lambdas from early layers lower than those from late layers? Because their impact on the output is reduced, so less of the impact has to be removed.

## Why I chose layer 11

I chose to add layer 11 to the output latents because I performed a simple ablation where I tried each layer output once (except for the last one, because why would I add the last layer output to the last layer output?), and it showed the following curve:

![Final losses when adding different layer output latents to the last layer output latent](images/final-val-losses-add.png)

All but layer 1 (the second layer) reduce the final validation loss, but layer 11 (the 12th layer) reduces it the most, so I chose it.

One possible explanation for why that is has to do with another aspect of the modded-nanogpt architecture: there are skip-connections between these layers: 6&rarr;9, 4&rarr;10, and 2&rarr;11. This means that at layer 11, we effectively pass (a weighted sum of) two layers to the output, which are very far apart. Substracting the layer 11 outputs thus effectively substracts the contributions from multiple layers.

And if the layer representations are very similar between adjacent layers, then that explains why adding more layers together at the output latents doesn't really help: with layer 11, we already have a good representation of all the layers' outputs available for substraction.

## Other experiments

I originally started these experiments trying to overcome the softmax bottleneck.

> The softmax bottleneck is an effect of the model dimension usually being much, much smaller than the vocabulary size (in our case, 1024 vs. >50,000). Because the expressivity of the latents is much more constrained than that of the final probability distribution, the LLM cannot independently optimize the output distribution for all contexts. It must therefore learn to tie multiple contexts&mdash;ideally, very similar ones&mdash;together. To do so, it typically "blends" the distributions, making them less crisp.

A possible mitigation is to increase the output latent dimension more gradually to the vocabulary size, by chaining multiple linear layers. However, that's expensive, and multiple linear layers without a residual connection hurt gradient flow. Other mitigations include the [Mixure of Softmaxes](https://arxiv.org/abs/1711.03953), [non-linear language heads](https://proceedings.mlr.press/v97/ganea19a.html), or [bilinear](https://arxiv.org/abs/2305.03452) language heads.

I experimented with a different technique.

### First attempt at mitigating the softmax bottleneck

I thought I might just concatenate the output of a previous layer to the final output latent and instantly double the dimension from which we project into the vocabulary. I tried this for every single layer's output (including the last layer's), plus the input embedding and an extra embedding, and noted the final validation loss for each of the runs:

![Final val losses: concatenating layer outputs](images/final-val-losses-concat.png)

There are clearly several layers which reduce the validation loss after 5690 steps significantly.

However, an obvious objection to these results being meaningful is that the language head doubled in size (we project from a vector twice the size as before into the same vocabulary size). Two facts from this plot speak against parameter count being the only reason for the improved performance:

1. Simply adding the input embeddings, or even an extra embedding layer just for the job, *increases* validation loss, even though they do contain some information about the input (and adding them in a weighted sum to the residual stream at every layer does reduce the validation loss!) &rarr; pure parameter count cannot be the only reason for the improved performance
2. Concatenating the output of layer 15 to the output latent&mdash;which *is* the output of layer 15&mdash; reduces the final validation loss a bit (I think it's kind of similar to the [Mixture of Softmaxes](https://arxiv.org/abs/1809.09296), except less efficient); but earlier layers make a much larger difference. Again, the actual layer reuse matters, either through increased information content at the output, or a reduced effective depth that the gradient has to travel at the layer we concatenate

Of course, a third piece of evidence that the layer reuse is an important component of the reduced validation losses comes from the final record: adding a previous layer's activations to the output latents adds exactly two parameters (the weights of the weighted sum between the vectors), which is nothing. Therefore, something else must be responsible for the improved performance.

Let's compare this early attempt (using concatenation) to the final record attempt (using a weighted sum):

![Concat vs add final val loss](images/concat-vs-add-final-loss.png)

A few observations:

- The minimum loss is achieved at different layers: 11 vs 12. This is likely just due to noise in the training process
- Concatenation of the middle layer outputs *increases* final validation loss, while adding them decreases it; that's strange and I don't have an explanation for it
- Concatenation does outperform the weighted sum significantly at some layers in terms of final validation loss

Considering that last fact, why is not this my official record? Well, let's look at the losses *over time*:

![Validation losses: concat layer 12 outputs](images/2312-val-losses.png)

The loss reaches 2.92 only after ~1506 seconds, which is far later than the ~1379 seconds for the actual record, so the extra parameters just make this so damn slow that it isn't worth it in this setting.

However, in a setting where the language head makes up far fewer of the parameters, it might be worthwhile to try this again. Since such a setting is, as far as I know, a setting where the model is very large, I cannot try it.

### Second attempt at mitigating the softmax bottleneck

In the past, I had tried simply adding the input embeddings x00 to the output latent in a weighted sum. However, despite the fact that doing the same at the residuals before the last layer decreases loss, this increased the final validation loss, and fairly significantly at that.

This made me think that the output of the final transformer layer shouldn't be touched; or at least that keeping the final transformer layer between any modification to the residual stream and the final output helps a lot.

With that in mind, I tried concatenating layer 12'th outputs to the residual *before* the final MLP (though after attention). To make up for the vastly increased parameter count that results from this change, I reduced the expansion factor from 4 to 1, which meant that the parameter count was constant.

However, this caused me some problems. Back then I thought that memory was somehow the issue, so I used a linear layer to project the latents from layer 12 down to size 128. This way, the concatenated layers are only slightly larger than the non-concatenated ones. However, the final loss was 2.922, so not below the limit; and it still took more time per step. Therefore, I dismissed this line of work.
