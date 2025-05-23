# Infinite Tool Use

An LLM should never output anything but tool calls and their arguments.

The tools hold the specific, instantiated state of what the model is doing and its goals, while the model itself holds only the information it requires for its immediate task and some additional context, leading to specialization between the LLM and its tools.

In other words, exclusively working through tools allows models to externalize large parts of their intelligence to more efficient, domain-specific programs.

Table of contents:

- [Examples](#examples)
  - [Text Editing](#text-editing)
  - [3D Generation](#3d-generation)
  - [Video Understanding](#video-understanding)
- [AI safety](#ai-safety)
- [Thoughs on Training](#thoughts-on-training)
- [Thoughts on Architecture](#thoughts-on-architecture)
- [Conclusion](#conclusion)

## Examples

The benefits of infinite tool use are best understood via examples.

### Text Editing

Here's how I wrote this article *so far*: I had an idea and wrote it down in a few bullet points. Then, I wrote the introduction. While doing that, I jumped to the end of the article, added a few more bullet points, and edited others. I started writing this section, interrupted it by writing down an idea about the architecture of such models, then came back here; realized that I should re-write this section, started doing that, edited the introduction to fit, went back to the re-write, and here we are. I'm not even half-way done with the article and I'm sure I already forgot several steps that I took.

Now contrast that with the way an LLM currently writes: It generates text forward-only. (Almost) no matter how good it is, it will make mistakes, especially in out-of-distribution (OOD) domains.

And while we can train it to backtrack and correct those mistakes in the form of **reasoners**, the mistakes themselves are baked into its output, and thus into both its own context window and the user answer. The latter is a problem because it makes it hard to produce long, correct outputs, the former because it is confusing to the model itself if the output gets very long.

Additionally, forward-only generation makes **multi-resolution generation** much more difficult: I as a human can create hundreds of versions of the same article; edit a sentence here and there, write down an idea as a bulletpoint, delete something dumb, turn a bulletpoint into a full section, etc.; in other words, I can interleave actions at different levels of specificity. Imagine how confusing it would be to hold all those edits in memory at once!

Editing through external tools allows for explicit, selective forgetting. LLMs on the other hand either need to generate from most general to most specific in order&mdash;a very limiting way of multi-scale generation compared to tool-use&mdash;or generate a confusing mess of edits and re-edits and deletions that aren't true deletions; or re-generate the entire output for every single edit; or just generate the final version all at once.

More generally, **extremely long contexts** are difficult to manage for LLMs, but might be required for very complex tasks.

Methods like [Entropix](https://github.com/xjdr-alt/entropix) try to work around these issues by dynamically adapting token-sampling parameters like temperature, by branching and merging on demand, and even backtracking, all based on an external measurement of the model's entropy. Good sampling strategies will be valuable no matter what, but leaving the editing decisions to the model itself is likely a more scalable approach, as demonstrated to a degree by current reasoners.

Speaking of reasoners, Chain of Thought (CoT), often trained via Reinforcement Learning (RL), is a different approach for allowing the models to fix these issues, but it doesn't address the problem of mistakes in the final answer, even if the CoT contains all the information necessary to produce a great final output. Of course, we can interleave CoT and user-output; but then we still commit to part of the final output early.

The final (and correct) step of this evolution is to simply allow the model to continually improve the final answer *before* dumping it on the user. Give it access to a full text-editor that is controllable through special text-commands, and see many benefits:

- Multi-abstraction-scale text generation
- Effortlessly interleaving those abstraction levels
- Backtracking via editing

Additionally, the use of SSMs in this context would enable the following:

- Extremely long edit and re-edit sessions without exploding costs
- Selective forgetting of obsolete information over the course of editing
- Immediate availability of the most up-to-date version of the desired output.

And the potential issue of going off-course if solved by simply refreshing the model's memory about fine-grained details (specific sections, sentences, words, what the goal of the whole process is, ...) through tool-use.

To be clear, this wouldn't prevent the model from generating easy answers in forward-only mode. If the LLM wants to directly answer a user without going through an editing process, they can do the equivalent of typing out a quick response and immediately clicking "send" within the tool.

### 3D Generation

3D generation, and the other examples listed below, face the same issues in normal LLMs, and can expect the same benefits from tool use, as pure text generation.

So what would a similar tool look like for 3D generation? CAD libraries exist for Python, and I do believe that there are programming languages for several Game Engines. Therefore, an LLM could create 3D objects through code. To do so, the model should have these tools available to it:

- A way to look at the generated object, given tools to:
  - Zoom in and out
  - Rotate the object
  - Shift the object
  - And, obviously, look at the resulting 2D projection of the 3D object
- A way to think about the object
  - This could just be another text-file as discussed in the [Text editing](#text-editing) section for taking notes
  - The model's context window itself (most thoughts should not be persistent, and the ones that should be can be written down in the note-taking-file)
- A way to edit and run the code itself
  - In other words, another file in a text-editor (or multiple files)

This would bring the following advantages:

- Generation of arbitrary-sized objects is possible.
  - This is currently not possible with text-to-3D models because in 3D-space, context windows explode when generating voxels etc.
  - But with a CAD-library&mdash;or numpy, as OpenAI's o3 is apparently doing&mdash;and the aforementioned tools, gradual generation of the object over many cycles of improvement becomes possible; in other words, human workflows are enabled
- All the advantages of the text-editing tool discussed above are available to the model

### Video Understanding

A full-attention LLM is un-usable for days-long videos because it's way too inefficient. A pure SSM is un-usable for the task because it cannot attend to enough of the video. But any LLM with a finite context window but *with tools* (and training to use them) can re-watch whatever part of the video it needs to understand what it has to, write down, edit, and revisit running notes, and more, without exploding costs. This makes it the obvious choice.

## AI Safety

Seeing the full editing process (with version control, potentially available to the LLM as well) is bound to be fascinating. More importantly, it has safety advantages.

This can be seen by analogy to current reasoners (which are complementary to the infinite tool use paradigm, but also a proto-version of it): If the task is hard, the model must make maximum use of the tools at its diposal, which include the CoT. Since deception adds additional complexity to the CoT, it further complicates the model's work, so if its capabilities are saturated, it will communicate as clearly as possible to itself within the CoT.

The same is true for LLMs with infinite tool use: when trained on sufficiently difficult tasks, they must use the tools at their disposal with clarity and good structure. Therefore, training them on sufficiently difficult tasks with infinite tool use will likely make their outputs more faithful and more legible.

## Thoughts on Training

Obviously good training data is needed. The challenge is that infinite tool-use requires truly agentic behavior: Understanding goals in detail, choosing tools, spotting and correcting mistakes, and so on.

The obvious solution is to just scale RL. This might seem difficult to impossible to do with unlimited context length.

However, using LLMs with a limited context window and interacting only through tools means that there is likely no need to actively train for infinite context length, if we train to recover from mistakes & edit from many different starting points. LLMs with limited context window (SSMs, sliding window attention, ...) being forgetful means that just training fairly long context windows from diverse start and end points will probably generalize to infinite context windows.

## Thoughts on Architecture

For architecture, I'm open to all possibilities; [RWKV](https://www.rwkv.com/), [Mamba](https://arxiv.org/abs/2312.00752), [xLSTM](https://arxiv.org/abs/2405.04517), [Titans](https://arxiv.org/abs/2501.00663), [Test-Time-Training](https://arxiv.org/abs/2407.04620), simple attention with a sliding window.

I'm also open to using hybrids. One version that might make sense for infinite tool use is the inverse or normal hybrids (though I'm also open to those). Normal hybrids typically use a few SSM layers followed by a full (causal) Attention layer (often without positional information). For infinite tool use, it might (might!) make more sense to reverse that: several layers of sliding window attention for a high-precision but localized view of the sequence, followed by an SSM layer that provides a much more abstracted but longer range view of the input. To be clear, I'm not at all sure about this, and this paragraph exists mostly for the fun of speculation,

This section is mostly meant to stress the importance of a constant inference budget per token independent of context window size (or at least one that is limited as in sliding window attention).

## Conclusion

The tool-use paradigm is in full swing already&mdash;o3 by OpenAI, agentic RAG models by Pleias, etc.&mdash;but still limited to very short contexts, and to only parts of the model output. I propose performing all interaction with the external world, be that users, a computer, or another LLM, through tool-use, and to scale that tool-use to ever-longer contexts using models that trade imperfect recall of the entire sequence (sliding window attention, SSMs, ...) for constant (or upper-limited) per-step cost.

## Acknowledgements

Thanks to [stochasm](https://x.com/stochasticchasm) for proof-reading the article and for fruitful discussions.

## Citation

```bibtex
@misc{snimu2025infinitetooluse,
    title={Infinite Tool Use},
    author={Sebastian Nicolas Müller},
    year={2025},
    month={04},
    url={https://snimu.github.io/2025/05/23/infinite-tool-use.html}
}
```
