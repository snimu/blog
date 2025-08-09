# modded-nanogpt: Looking at value embedding lambdas, U-Net lambdas, and x0-lambdas

The [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/tree/master) [medium-track](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py) makes use of multiple tricks to improve performance, which rely on learned scalar values&mdash;the lambdas&mdash;for mixing two vectors:

- U-Net: at three points in the model, the residual is modified by an earlier value of the residual via a weighted sum
- X0-Mixin: At every layer, after the U-Net skip (if it occurs at that layer) but before Attention and the MLP, the current residual and the original input embeddings are mixed in a weighted sum
- Value-Embeddings: In each Attention layer, value embeddings (extra embeddings of the input tokens) are added to the Value vector before the actual Attention operation is applied

I have measured and plotted the weights of these weighted sums. In this article, I will go through all three tricks, first explaining them shortly and then plotting the learned lambdas over training, and over the layers where it makes sense. Warning: these are all from a single training run, so variations are possible.

This is for the code as of August 9th, 2025, corresponding to [this record log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/042225_GPT2Medium_Record8/075_640429f2-e726-4e83-aa27-684626239ffc.txt).

## U-Net Lambdas

The model mixes the activations at position `a` into those at the later position `b` in the following way: `x_b = x_b + lambda_ab * x_a`. This is very similar to a U-Net, so I'll call them the U-Net Lambdas.

They shorten the gradient path from the output to the input (they are residual connections skipping multiple layers), and are initialized to 1.0, which means that the model is initialized to have a short effective depth, which only grows over time, and that the value of the lambda at the end of training is equivalent to a skip percentage during inference.

The following layers are connected:

- 2 &rarr; 11
- 4 &rarr; 10
- 6 &rarr; 9

This makes for three scalar values that are optimized for the entire model. Here is their development over the course of training:

![U-Net Lambdas](images/unet-lambdas.png)

I find it interesting that the amount of layer-skipping at the end of training is very significant:

- Layers 0-2 and 11-15 are never skipped
- Layers 3-10 are skipped by ~39%
- Among those, layers 4-9 are skipped by 50%
- And then among those, layers 6-8 are skipped by ~31%

The middle layers contribute relatively little to the output! Of course, that can be made up for by scaling the norm of the middle-layer outputs (which I unfortunately didn't measure), but my guess is that it's meaningful.

> Sidenote: making these lambdas data-dependent could allow the model to dynamically choose to make more use of the middle layers for some problems, and less for others. While that wouldn't save compute in this architecture, it would give the model more algorithmic flexibility. Might be worth a try, though it's likely too expensive to make for a record, and may not even help.

## X0-Lambdas

At every layer of the model, there are two scalars used for mixing the input embeddings `x0` into the residual stream in a weighted sum: `x = x_lambda * x + x0_lambda * x0`.

I don't fully understand the purpose of this modification to the standard transformer architecture. It definitely makes for a very short gradient path to the input, and also provides a gradient to the input embeddings from every layer, speeding up their training (in the backward pass, the layers effectively act as data augmentations to the gradient, which allows for longer training with the same data and will thus be helpful for updating the embeddings).

I have recorded the x0-lambdas for all layers over the entirety of training. Plotting all of these is very ugly, so here for my main plot I only show the final value of the x0-lambdas over the model's layers, normed so that they sum to 1 (because we care about their relative weight):

![x0-lambdas, normed](images/x0_lambdas_normed.png)

I can see three points of interest:

First, the lambdas are almost identical in layer 0. That makes sense, because in layer 0 (and only in layer 0), `x == x0`.

Secondly, at layer 8, the sign is flipped. That isn't very meaningful except that the parameter had to learn to cross 0. What is interesting is that the x0-lambda is noticeably higher here than in the other layers (except for the last). That may be chance, but it may also be a result of the previous layer skipping the attention layer. It would seem that layer 7 contributes less to the model than the other layers, and is thus overwritten more strongly with x0. That's pure speculation though.

Lastly, the last layer has a very large x0-lambda, almost 80% of the input to layer 15 consists of the input embeddings (which is again achieved by the x-lambda crossing 0 and becoming negative). That makes me think that the final calculation is a simple embeddings calculation, like the stereotypical "king - man + woman = queen"; the transformer simply calculates the difference between the input and target embeddings. That is a common view, but it's shown strongly here (and it also demonstrates the related view that the transformer is a learned gradient estimator).

But what it also does is reveal a second purpose of the x0-lambdas: they allow the model to compare the residual&mdash;meaning, in abstract terms, the vector it plans to add to the input embeddings&mdash;to the input to which it will be added, and adjusting accordingly. That's a bit imprecise, but it helps my intuition.

For those who want all the details, you can extend the below section to see the x0-lambdas over the layers *and* over the course of training.

<details>
<summary>Per-layer x0-lambdas over training</summary>

First the normalized values:

![x0-lambdas normed, layers 0-3](images/x0_lambdas_0-3_normed.png)
![x0-lambdas normed, layers 4-7](images/x0_lambdas_4-7_normed.png)
![x0-lambdas normed, layers 8-11](images/x0_lambdas_8-11_normed.png)
![x0-lambdas normed, layers 12-15](images/x0_lambdas_12-15_normed.png)

And, for the sake of completeness, here are the un-normed values:

![x0-lambdas un-normed, layers 0-3](images/x0_lambdas_0-3.png)
![x0-lambdas un-normed, layers 4-7](images/x0_lambdas_4-7.png)
![x0-lambdas un-normed, layers 8-11](images/x0_lambdas_8-11.png)
![x0-lambdas un-normed, layers 12-15](images/x0_lambdas_12-15.png)

</details>

## Value-Embeddings Lambdas

The value embeddings are embedding layers beside the one that forms x0. Their embeddings are added to the Values in the causal self-attention block right before flexattention is applied: `v = ve_lambdas[0] * v + ve_lambdas[1] * ve.view_as(v)`

However, that is only the case in the first and last three layers. In between, only `ve_lambdas[0]` is used to scale the Values, while `ve_lambdas[1]` remains unchanged over training. And at layer 7, there is no attention, so there are no such lambdas.

To add more complexity, the value embeddings of the first and last three layers are shared; so layers 0 and 13, layers 1 and 14, and layers 2 and 15 share the same value embeddings. This saves parameters and ensures that there is a short gradient path from the loss to the value embeddings. While additional gradient comes from the first layer which are the farthest away from the input, there is always gradient from the three layers nearest to the loss, too.

I admit that I only have some very weak intuitions for why value embeddings help. I've long held that token embeddings are a way to hold training-set wide statistics about the byte-sequence represented by the token, so I'm guessing that they add a way for the model to store more static per-token statistics which are helpful. However, that's 100% speculation and you shouldn't take it too seriously, I'm just "thinking aloud" (in my writing).

I will again plot the final values of these lambdas over the layers; however, this time both the absolute and the relative values are of interest. The absolute values are specifically interesting for the layers in which there are no value-embeddings, but the v-lambda is sitll used to scale the values. The normed values are particularly interesting (at least in my eyes) for the layers where values embeddings are applied, and we want to see how strongly they are mixed in relative to the values.

So here is the plot with the absolute values:

![ve-lambdas, absolute](images/ve_lambdas.png)

Some observations about the layers without value embeddings:

- The model really likes to scale the attention-values by around 6-7
- There seems to be a trend to do this more strongly in the later layers, but it's weak and I don't know if it's meaningful
- The ve-lambdas at these layers of course stay unchanged throughout training, because they are never used

Some observations about the layer with value embeddings:

- The v-lambdas always increase, though not as much as without value embeddings
- The ve-lambdas stay very low, except in the last layer
- The ratio of v-lambdas to ve-lambdas is much higher in the first than the last layers; the value embeddings seem to have little impact on the first few layers (numberially, they could still stabilize training, or add just enough to make a difference, or whatever)

To make more sense of that last point especially, let's look at the normalized ve-lambdas:

![ve-lambdas, normed](images/ve_lambdas_normed.png)

Three clear groups emerge:

1. Layers 0-2, which are affected very weakly by the value embeddings (by only around 10%)
2. Layers 13 and 14, which are affected fairly strongly by the value embeddings (by around 25%)
3. Layer 15, which uses its value embeddings almost as strongly as its input from the residual stream (with around 45% intensity)

The last point is very interesting, because as we saw before, the last layer already mixes the original embeddings into the residual stream at its input with about 50% magnitude. That's a lot of fixed values that don't depend on the token-order at all!

Finally, below you can expand a section to see the full details over training, for those who like to look at plots all day.

<details>
<summary>Per-layer ve-lambdas over training</summary>

First the normalized values:

![ve-lambdas normed, layers 0-3](images/ve_lambdas_0-3_normed.png)
![ve-lambdas normed, layers 4-7](images/ve_lambdas_4-7_normed.png)
![ve-lambdas normed, layers 8-11](images/ve_lambdas_8-11_normed.png)
![ve-lambdas normed, layers 12-15](images/ve_lambdas_12-15_normed.png)

And, for the sake of completeness, here are the un-normed values:

![ve-lambdas un-normed, layers 0-3](images/ve_lambdas_0-3.png)
![ve-lambdas un-normed, layers 4-7](images/ve_lambdas_4-7.png)
![ve-lambdas un-normed, layers 8-11](images/ve_lambdas_8-11.png)
![ve-lambdas un-normed, layers 12-15](images/ve_lambdas_12-15.png)

</details>

## Summary

I've ran the modded-nanogpt medium track training once (so don't overindex on this) and plotted all the scalar values that are being trained. Hope you enjoyed.
