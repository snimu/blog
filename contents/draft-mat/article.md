# Question: do people re-use embedding layers?

I have two ideas for re-using embedding layers, and I'm wondering if somebody has tried them already. I will simply present these ideas and let the question hand in the room.

## Idea 1: Re-use embedding- and unembedding layers

- Take embedding and unembedding weights from existing model (like Llama-3-1b)
- Use them to embed the dataset input using the embedding layer, and compute the pseudo-inverse of the unembedding layer on the targets
- Then, train a new model purely in embedding space
- This saves memory during training
- It also jumpstarts training, because the embeddings are already semantically meaningful
- Question: would that limit the expressivity of the new model? If it's trained on a completely different dataset, will applying the unembedding layer lead to a good probability distribution over the vocabulary?
- Problem: embedding dimension fixed

## Idea 2: Make them Matrioshka layers

... explain ...

What can be done with this?

- What is described above, but for more embedding dimensions
- Also, distillation on the output hidden states instead of the tokens (potentially with post-training phase where the embedding, unembedding, and language model are trained on normal CE loss)
