# Sorting as a Universal Task

Sorting shuffled data is a great verifiable task for RL, and by extension, a good eval benchmark. In this article, I want to boost it so that it will be used more.

*What do I mean by sorting?* I mean taking a list of items&mdash;pages in a book or technical document, images from a video, audio clips, etc.&mdash;and shuffling them. Then, you let an LLM undo that shuffle.

## What are the advantages of this task?

There's a long list:

1. It is automatic and verifiable, as long as data to be shuffled is available.
2. With the right data, it requires true understanding of the contents.
    - Sorting video frames requires understanding movement; sorting the steps in an IKEA assembly manual requires spatial understanding (if the step number is removed from the data); and so on.
    - Of course, you need to make sure to use the correct data (don't include page numbers if you want to sort the pages, don't include time-related metadata if you want to sort video frames, etc., except as a baseline).
    - There is also data that can be sorted in multiple ways; this has to be taken into account.
3. The task can be defined for any modality: shuffle text, images, objects in images, video clips, etc.
4. The task difficulty can be controlled by the amount of shuffling. While global, random shuffling is possible, it's easy to instead shuffle with a pre-defined degree of locality (through random kernels or other methods).
5. The task difficulty can be controlled by the exact data that is used: modality, chunk-size, content, length, etc.
6. It is easy to set up: you only need to provide data, and define how it is split up (in other words, do you shuffle pages, or paragraphs, or images, ...?).
7. Because of that, it is resistant to contamination: you can always find a new piece of data that is not in the training set, and test with that.

## Open questions

How does the data look like?

- Do we have a list of tuples `[(<random id>, <datapoint>), ...]` and ask the model to return the sorted ids?
- Do we ask the model to re-produce the data itself, but in the correct order? That would add an additional task (which may be easy or very hard, depending on the data).

If companies use this for evals, how do we make the benchmark comparable between models?

- The problem:
  - A company trains a model, and evaluates it on a shuffled test set.
  - Later, another company does the same
  - That second company is at risk of data contamination, because the test-set might have leaked in the mean-time.
  - If the test-set hasn't leaked, it remained private; which means nobody can interpret or trust the original eval results.
- A proposal for a solution:
  - Each company evaluates on a new set of data and tasks.
  - Then, they release those tasks and the data, along with the eval resutls.
  - Now, when the next company wants to evaluate their own model, they can 1) eval on the same data, 2) use their own tasks and compare openly to the other company's model
  - This would ensure meaningful comparisons, and trust in the results.
- Another problem:
  - Companies might not want to do that
  - I don't know how to solve this issue

Whether it works as a benchmark or not, it's a great task for RL.

## Final remark

I've seen people use this as a benchmark before. I'm not claiming novelty here, I just want to stress how useful sorting shuffled data can be made.

## Citation

```bibtex
@article{snimu2025sorting,
  title={Sorting as a Universal Task},
  author={Sebastian Nicolas Müller},
  year={2025},
  month={Feb},
  url={https://github.com/snimu/blog/blob/main/contents/sorting-as-a-universal-task/article.md}
}
