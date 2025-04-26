# Infinite Tool Use

I am bullish on state-space models (SSMs) because I believe that ***the path to AGI lies in infinite context length***.

This is because infinite context length allows for human-like tool-use, which in turn allows models to externalize their intelligence; the tools hold the specific, instantiated state of what the model is doing, while the model itself holds an abstract representation of that state and its goals as well as the little bit of instantiated data it currently needs, leading to a specialization between tools to the task at hand. This means that both the model and its tools can make use of their comparative advantage. It is, I believe, what humans do, and for cost reasons, it cannot be done using self-attention.

So let's take this paradigm to its logical conclusion: ***I posit that an LLM should never output anything but tool calls and their arguments.*** Not only is tool-use a multiplier of LLM capabilities, it also alleviates the necessity for the LLM to have perfect access to the full context, making SSMs an attractive option.

To drive home these points, I will go into detail about an example of infinite context use-cases: text generation. Then, I will quickly touch on more tasks in which tool use with infinite context is helpful, and discuss some thoughts on the architecture and training of such models.

## Text editing

Here's how I wrote this article *so far*: I had an idea and wrote it down in a few bullet points. Then, I wrote the introduction. While doing that, I jumped to the end of the article, added a few more bullet points, and edited others. I started writing this section, interrupted it by writing down an idea about the architecture of such models, then came back here; realized that I should re-write this section, started doing that, edited the introduction to fit, went back to the re-write, and here we are. I'm not even half-way done with the article and I'm sure I already forgot a bunch of steps that I took.

Now contrast that with the way an LLM currently writes: It generates text forward-only. (Almost) no matter how good it is, it will make mistakes, especially in out-of-distribution (OOD) domains. And while we can train it to backtrack and correct those mistakes, the mistakes themselves are baked into its output, and thus into both its own context window and the user answer. The latter is a problem because it makes it hard to produce long, correct outputs, the former because it is confusing to the model when the outputs are long.

Additionally, forward-only generation makes multi-resolution generation much more difficult: I as a human can create hundreds of versions of the same article; edit a sentence here and there, write down an idea as a bulletpoint, delete something dumb, turn a bulletpoint into a full section, etc.; in other words, I can interleave actions at different levels of specificity. Imagine how confusing it would be to hold all those edits in memory at once!

Editing through external tools allows for explicit, selective forgetting. LLMs on the other hand either need to generate from most general to most specific in order&mdash;a very limiting way of multi-scale generation compared to tool-use&mdash;or generate a confusing mess of edits and re-edits and deletions that aren't true deletions; or just generate the final version all at once.

Methods like Entropix try to work around these issues by dynamically adapting token-sampling parameters like temperature, by branching and merging on demand, and even backtracking. All those are based on an external measurement of the model's entropy. Good sampling strategies will be valuable no matter what, but: Why not leave these decisions to the model itself?

Chain of though (CoT) is a different approach for allowing the models to fix these issues, but it doesn't address the problem of mistakes in the final answer, even if the CoT contains all the information necessary to produce a great final output. Of course, we can interleave CoT and user-output; but then we still commit to part of the final output early.

The final (and correct) step of this evolution is to simply allow the model to continually improve the final answer *before* dumping it on the user. Give it access to a full text-editor that is controllable through special text-commands (it might make sense to standardize those so that the tools can be shared between model providers, but that's neither here nor there), and see many benefits:

- Multi-scale text generation
- Interleaving of those scales
- Backtracking with selective forgetting of obsolete information

Additionally, the use of SSMs in this context would enable the following:

- Extremely long edit and re-edit sessions without exploding costs
- Forgetting irrelevant information over the course of editing
- While refreshing the model's memory about fine-grained details (specific sections, sentences, words, what the goal of the whole process is, ...) through tool-use

And it wouldn't prevent the model from generating easy answers in forward-only mode, by doing the equivalent of tying out a quick response and immediately clicking "send".

## Other examples

There are many things that could be improved by the framework of infinite tool-use: 3D generation, video understanding, robotics, etc.

### 3D generation

CAD libraries exist for Python, and I do believe that there are programming languages for several Game Engines. Therefore, an LLM could create 3D objects through code. To do so, the model should have these tools available to it:

- A way to look at the generated object, given tools to:
  - Zoom in and out
  - Rotate the object
  - Shift the object
- A way to think about the object
  - This could just be another text-file
- A way to edit and run the code

The latter two would of course use the same tools as discussed in the [Text editing](#text-editing) section.

This would impart the following advantages:

- Generation of arbitrary-sized objects is possible.
  - This is currently not possible with text-to-3D models because in 3D-space, context windows explode when generating voxels etc.
  - But with a CAD-library&mdash;or numpy, as OpenAI's o3 is apparently doing&mdash;and the aforementioned tools, gradual generation of the object over many cycles of improvement becomes possible; in other words, human workflows are enabled
- All the advantages of the text-editing tool discussed above are available to the model

### Video understanding

A full-attention LLM is un-usable for days-long videos because it's way too inefficient. A pure SSM is un-usable for the task because it cannot attend to enough of the video. But an SSM *with tools* can re-watch whatever part of the video it needs to understand what it has to, write down, edit, and revisit running notes, and more, without exploding costs. This makes it the obvious choice.

### Robotics

All the advantages mentioned before a applicable to robotics: interleaved video (and audio etc.) understanding, planning, and acting, all over infinite context lengths, is much closer to what humans do than pure forward-generation by context-window-constrained, attention-based, forward-only VLMs currently do. I'm not convinced that it is enough without continual learning, and the processes aren't truly parallel anyway, but I do believe that it's a step in the right direction.

## AI safety

Seeing the full editing process (with version control, potentially available to the LLM as well) is bound to be fascinating. More importantly, it has safety advantages: since the model relies heavily on the external tool, which is legible to humans because it's mostly text, lying is disincentiviced because it makes the actual task harder by forcing the model to keep track of far more variables, thus increasing the success-rate of honest LLMs relative to dishonest ones.

## Thoughts on training

Obviously good training data is needed. The challenge is that infinite tool-use requires truly agentic behavior: Understanding goals in detail, choosing tools, spotting and correcting mistakes, and so on. The obvious solution is to just scale RL; but can we find some good seed-data? I believe so.

For code, GitHub is great. We can turn the current state of the code-base, and an issue, all the commits leading to a PR being merged, and the PR itself&mdash;all in the order in which they were created&mdash;into a single sequence and pre-train on it. This would still not fully emulate infinite tool-use, because it doesn't include calls to load pieces of code for review into the context, and similar issues. It's also often missing discussions that happen outside of the issues and PRs. But it's a good starting point.

The same is true for something like Google Docs, which have an edit history.

For 3D generation, I imagine such data to be much harder to come by, so a stronger focus on RL is required.

However, using SSMs and interacting only through tools means that there is likely no need to actively train for infinite context length, as we train to recover from mistakes & edit from many different starting points. And that is my main takeaway: SSMs being forgetful means that just training fairly long context windows from diverse start and end points will probably generalize to infinite context windows.

## Thoughts on architecture

For architecture, I'm open to all possibilities; [RWKV](https://www.rwkv.com/), [Mamba](https://arxiv.org/abs/2312.00752), [xLSTM](https://arxiv.org/abs/2405.04517), [Titans](https://arxiv.org/abs/2501.00663), [Test-Time-Training](https://arxiv.org/abs/2407.04620), etc. I'm also open to using sliding-window self-attention with a really large sliding window (100k tokens or whatever you can take) every few layers to give the model high-resolution access to the immediate context, though then we really have to train on context lengths much longer than the sliding window to train the model to make proper use of the SSMs.

I say this to stress that the important point is a constant inference budget per token independent of context window size (or at least one that is limited as in sliding window attention).

## Conclusion

The tool-use paradigm is in full swing already&mdash;o3 by OpenAI, agentic RAG models by Pleias, etc.&mdash;but still limited to very short contexts, and to only parts of the model output. I propose performing all interaction with the external world, be that users, a computer, or another LLM, through tool-use, and to scale that tool-use to ever-longer contexts, which would necessitate models that are linear in complexity over the sequence length.

## Citation

```bibtex
@misc{snimu2025infinitetooluse,
    title={Infinite Tool Use},
    author={Sebastian Nicolas Müller},
    year={2025},
    month={04},
    url={https://snimu.github.io/2025/04/26/infinite-tool-use.html}
}
```
