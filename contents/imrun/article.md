# Imrun: speedrunning image understanding

I'd like to have a speedrunning repo like [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), but for image understanding.

I want this repo to have maximum flexibility. It should be usable for testing many different methods. Those methods are not necessarily comparable through loss though: how do you compare JEPA loss to diffusion loss? You don't. So instead, common evaluations are needed that test many different capabilities. However, many of those evaluations are fairly expensive and shouldn't be run every few steps, so the equivalent of "lowest time to fixed val loss wins" from modded-nanogpt won't work for comparisons. In a sense it's not quite speedrunning, but it fulfills the same purpose.

Here's my proposal for how a winner should be determinded instead: Train for a fixed time (+/- n seconds), then evaluate once. The best eval scores win. This has some advantages:

- The obvious one is that it deals with the two problems noted above: supporting a variety of methods, and give meaningful scores for all of them, while making the training resources comparable (fixed time on fixed hardware)
- Also, it allows us to have a large variety of evals and thus a much more nuanced view on the strengths and weaknesses of the different methods

## Evaluations

This is the most important part of the repo.

### Examples of evaluations

Below are some rough examples of possible evaluations that could be used.

#### Linear probe transfer

As done for [JEPA](https://arxiv.org/abs/2301.08243), train a linear probe (or a cross-attention probe as in [V-JEPA](https://openreview.net/forum?id=WFYbBOEOtv)) to perform some downstream task like Imagenet or Cifar100 classification.

This should probably be the central pillar of the eval setup, because it is well established.

What do we do when the different test-models produce different outputs? I'd say just let people use whatever output of the test model that they want to use as the input to the evals, and train the linear probes for a fixed amount of time on fixed hardware. Then, large hidden states will train the probes with fewer examples, making the comparison fair.

#### Image-similarities

To test a method's viability for producing image-embeddings, we can take any hidden state or output produced by the model in response to an image-input, and compare it between different images. Similar images should produce similar scores, dissimilar ones should produce dissimilar scores.

As a baseline, I would pick some image-embedding model like [ColPali](http://huggingface.co/vidore/colpali-v1.3) or [ColQwen](https://huggingface.co/vidore/colqwen2.5-v0.2) and compute the cosine-similarities between a bunch of imagenet images. Since those models have been trained on *much* more data than any of the speedrun models, they can be seen as ground truth. Their scores can also be computed offline, one time only, and then saved for comparison via evals.

Then, compute the hidden states and their similarity scores between the same image-pairs using the test-model. To make the similarity computations comparable, normalize the similarity scores of the baseline to between 0 and 1, and do the same to those of the test-model.

Now, if we assume that ColPali/ColQwen are very strong models, their similarity scores and those from our test-model should ideally be identical, and we can compute a loss.

#### Approximating viability for VLMs

To approximate the viability of the method for VLMs, we can simply train a VLM. Given some dataset containing image-text pairs, put the hidden states of the test model for the image&mdash;extracted from wherever you want in the model, and reshaped however you want&mdash;into the context window of a tiny transformer, then the text belonging to the image afterwards. Train on only the text, for a given number of seconds (so small hidden states can run for more steps). The minimum loss over a few runs is then a good measure of information density in the test model's hidden states. As the dataset and VLM can be tiny, this comparison should be fairly cheap.

The same could be done with other things, like training a tiny diffusion model with guidance from the hidden states, but I'm far less knowledgeable on that front, so it's not something that I would do from the start.

### General points about evaluations

A few points:

- The examples listed above are just a collection of early ideas, a lot of them won't be viable
- An important point is to have very different methods be comparable, which is why I don't like comparing them on their actual outputs; JEPA outputs are just so damn different from diffusion outputs, or CLIP/SigLIP adapter outputs
- On the other hand, some of the evaluations might be too inefficient. Is it really realistic to download the entirety of Imagenet for these evaluations? It's probably necessary to manually cache subsets of a few datasets so that they can be downloaded easily and quickly (preferably with the help of some download script)
- How do we compare models that use inputs from other models (like DINOv2 embeddings, or text embeddings for guidance) to models that don't? Just keep separate leaderboards?

## Methods to compare

A good idea would probably be to go through a bunch of [lucidrains](https://github.com/lucidrains) repositories and implement rough drafts & training scripts, setting up the evals based on them, then sharing the repo, evals, and early results and hope that others who are better at the specific methods will then jump in.

Some of the early methods I'd like to implements:

- [MAE](https://arxiv.org/abs/2111.06377)
- [JEPA](https://arxiv.org/abs/2301.08243)
- Some sort of diffusion model
- A simple CLIP model (together with whatever LLM fits into 8xH100s)

## Open questions

- Training data: do we use a fixed dataset, or do we let people iterate on that as well? I lean toward a fixed dataset, but I'm not completely sure yet
- Code structure:
  - The eval code must be separate from the training code, because there should be a training file per method.
  - In fact, it might make sense to have a separate directory per method: one for diffusion, one for flow matching, one for MAE, etc., and have all the record traces of that method in the respective directory
  - Each method should probably implement a function that loads the trained model and produces the hidden states in some format, though there is a tradeoff between how uniform this interface is compared to how hackable it is
