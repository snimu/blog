# Model stacking doesn't work

In my article about [Model Stacking](https://snimu.github.io/2025/02/18/model-stacking.html), I proposed a method for decentralized pre-training of models.

I have since performed extensive experiments that show that it doesn't work. I present these experiments below. Note: I've obtained these results several months ago and only got around to writing them down now, so I unfortunately don't remember every nuance of my motivations, and the writing will be fairly short.

## The method

The basic insight is that tying a language head (lm-head) to the token embedding layers&mdash;a somewhat common practice for saving parameters&mdash;should force the model to work in the same embedding space at the input and output. This would allow us to do the following:

1. Train model 1 using tied embeddings and lm-head
2. Train a differently initialized model, model 2, but with the same (now frozen) embeddings and lm-head
3. For inference, remove the lm-head from model 1 and the embeddings from model 2, and stack the two, which, according the the original idea, should work well together because their embedding spaces are aligned

In principle, this could be scaled to stack many model atop each other.

## Experiments

You can find my code here: [https://github.com/snimu/model-stack](https://github.com/snimu/model-stack)

It's based on [this old modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/train_gpt2.py), because 1) it uses tied embedding and unembedding weights, and 2) it's still a fairly simple model, which is probably good for stacking the models.

As my experiments, I simply followed the receipt from the previsous section, then compared the validation losses of the individual models and the stacked model.

### Model variations

When doing the experiments, I varied three things: the norms, how many of the layers are stacked, and model looping.

Norming the activations at the in- and output is important, because the loss on the final outputs includes a softmax, which means that the residual magnitude can differ between in- and output even if the embedding and lm-head weights are tied. I independently test norming the token embeddings (`norm_wte`), the inputs to the lm-head (`norm_lm_head`), and the residual activations in the transformer blocks between the two (`norm_inter_model`).

Not stacking all layers of the models is based on the observation that early and late layers are used to move embeddings into and out of the abstract space of actual thinking, to and from the concrete space of individual tokens. If we want to stack models, we should stack them in such a way that they share the abstract thoughts, not the concrete token-embeddings.

Looping the model like in [COCONUT](https://arxiv.org/abs/2412.06769) might teach the model to make use of its own outputs. If I do this to both models, and both work in the same embedding space at both in- and outputs, then model 2 might learn to make use of the outputs of model 1 at its input. To train these models with looping in mind during pre-training, I use the techniques described in my article [COCONUT: parallel pre-training](https://snimu.github.io/2025/01/12/COCONUT-parallel-pretraining.html) (Spoilers: this makes everything worse, so the method doesn't work).

As a baseline, I always stack the same model twice (when I don't, `unique_model` is true).

## Results

These are the column names:

- `val_loss_stack`: The validation loss of the model stack
- `val_loss_mean`: The mean of the validation losses of the models that are being stacked
- `use_first_layer`: Whether or not the first layer of the second model in the stack is used
- `use_last_layer`: Whether or not the last layer of the first model in the stack is used
- `norm_wte`: Did we use an embedding norm (in model training and the model stack)?
- `norm_lm_head`: Same but for the last transformer block, right before the lm head
- `norm_inter_model`: Should be called "intra-model" but whatever; it means norming the residual between the transformer blocks
- `unique_model`: If true, the two models in the model stack were trained from a different seed; if false, they are both the same model
- `coconut_every`: How many normal steps before each COCONUT-parallel-style update step?

### Full results

Here are the full results:

|   val_loss_stack | use_first_layer   | use_last_layer   | norm_wte   | norm_lm_head   | norm_inter_model   | unique_model   |   mean_val_loss |
|-----------------:|:------------------|:-----------------|:-----------|:---------------|:-------------------|:---------------|----------------:|
|            10.57 | False             | False            | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|            10.02 | False             | False            | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             8.28 | False             | False            | none       | rms_norm       | none               | False          |            3.30 |
|             6.48 | False             | False            | none       | rms_norm       | none               | True           |            3.28 |
|            12.85 | False             | False            | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.02 | False             | False            | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|             9.37 | False             | True             | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|             9.97 | False             | True             | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             7.48 | False             | True             | none       | rms_norm       | none               | False          |            3.30 |
|             7.01 | False             | True             | none       | rms_norm       | none               | True           |            3.28 |
|            14.78 | False             | True             | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.40 | False             | True             | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|            10.68 | True              | False            | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|            10.42 | True              | False            | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             8.28 | True              | False            | none       | rms_norm       | none               | False          |            3.30 |
|             8.31 | True              | False            | none       | rms_norm       | none               | True           |            3.28 |
|            11.25 | True              | False            | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|            10.35 | True              | False            | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |
|            10.37 | True              | True             | layer_norm | layer_norm     | layer_norm         | False          |            3.31 |
|             9.84 | True              | True             | layer_norm | layer_norm     | layer_norm         | True           |            3.30 |
|             6.92 | True              | True             | none       | rms_norm       | none               | False          |            3.30 |
|             7.65 | True              | True             | none       | rms_norm       | none               | True           |            3.28 |
|            11.44 | True              | True             | rms_norm   | rms_norm       | rms_norm           | False          |            3.31 |
|             9.97 | True              | True             | rms_norm   | rms_norm       | rms_norm           | True           |            3.30 |

The most important results is that the model stack is always worse than the individual models it's made up from, and by a significant margin.

### Results by layer use

|   val_loss_stack |   mean_val_loss | use_first_layer   | use_last_layer   | unique_model   |
|-----------------:|----------------:|:------------------|:-----------------|:---------------|
|             9.79 |            3.30 | False             | False            | True           |
|             9.79 |            3.30 | True              | False            | True           |
|             9.60 |            3.30 | False             | True             | True           |
|             9.60 |            3.30 | True              | True             | True           |
|             9.79 |            3.30 | False             | False            | False          |
|             9.79 |            3.30 | True              | False            | False          |
|             9.60 |            3.30 | False             | True             | False          |
|             9.60 |            3.30 | True              | True             | False          |

Clearly, whether the first layer of the second model is used or not makes no difference, but whether or not the last layer of the first model is used does: not using it noticably improves the performance of the model stack. This is independent of whether or not the model being stacked is unique.

### Results by norm

|   val_loss_stack |   mean_val_loss | norm_wte   | norm_lm_head   | norm_inter_model   | unique_model   |
|-----------------:|----------------:|:-----------|:---------------|:-------------------|:---------------|
|            10.06 |            3.30 | layer_norm | layer_norm     | layer_norm         | True           |
|            10.19 |            3.30 | rms_norm   | rms_norm       | rms_norm           | True           |
|             7.36 |            3.28 | none       | rms_norm       | none               | True           |
|            10.24 |            3.31 | layer_norm | layer_norm     | layer_norm         | False          |
|            12.58 |            3.31 | rms_norm   | rms_norm       | rms_norm           | False          |
|             7.74 |            3.30 | none       | rms_norm       | none               | False          |

The first thing that jumps out at me is that here, using two different models does make a difference: it's significantly better than stacking the same model on top of itself.

Other than that, using not `norm_lm_head` and no `norm_wte` is by far the best setting, which I remember surprising me a lot, because I expected it to be important that the activations at the model in- and outputs are aligned, the more the better. Accordingly, I expected Layer Norm to outperform RMS Norm, whic it did by a small margin.

### Results with parallel COCONUT

I did these with the best setting discovered before: no `norm_lm_head` and no `norm_wte`, and two different models in the model stack. In this case, I performed two to four runs per setting, and will simply show the mean validation losses below.

|   val_loss_stack |   mean_val_loss | use_first_layer   | use_last_layer   |   coconut_every |
|-----------------:|----------------:|:------------------|:-----------------|----------------:|
|             8.09 |            3.32 | True              | True             |          100.00 |
|             8.00 |            3.32 | True              | False            |          100.00 |
|             8.85 |            3.32 | True              | True             |           10.00 |
|             9.66 |            3.32 | True              | False            |           10.00 |

Some observations:

- Using model looping in the way I've proposed increases the validation loss of the trained model (that's another hypothesis disproven)
- Using model looping more often during training doesn't worsen the trained model significantly, but it does worsen the stacked model
- Using the last layer of the first model makes a larger difference when `coconut_every` is lower

All in all, this trick seems to make things worse rather than beter.

## What now?

I won't be pursuing the idea further.

Importantly, stacking and then post-training makes little sense, because there are better methods of franken-merging a model with itself and then post-training, or training for layer looping.

## Citation

...
