# \<Tags\> make them smart, \<Tags\> make them safe

I've read in muliple places that including metadata about a piece of text in tags around that text improves pre-training performance of LLMs. But those works, as far as I know, end their training runs on raw texts so that the model can be used without tags. I don't understand why that should be desireable.

In LLMs, there is a conflict between pre- and post-training: Pre-trained LLMs are simulators, but companies post-train them to be agents, which can make the LLMs worse world models but more economically useful. Tags resolve that conflict: just have the simulator simulate the agent by training it to behave a certain way, conditional on some tags.

The tags allow for a **separation of simulacra**, and that separation will be much sharper and much easier to control than when training and doing inference with pure text.

These advantages generalize to many domains: safety, continual learning, character design, and more.

To be clear, this is an extremely speculative post, and you should take everything I say with a large grain of salt. I will phrase a lot my claims very strongly, but even if the claims turn out to be directionally correct, effects may be small or nonexistent.

## What exactly do I mean by tags?

Quite simply, give the model metadata about the text it is trying to predict. For example. `<url>https://snimu.github.io></url>`, or `<date>2025-05-24</date>`.

I'd make the tags themselves (`<url>`, `<data>`, ...) special tokens, which has three advantages:

1. The tags cannot just be typed out and faked by users if you don't want that (for example, if you want to use tags to closely control model behavior in your App)
2. If you *do* want the tags to be changeable by the users, they will have a huge impact (they were trained throughout the entirety of pre-training), giving users a great amount of control
3. At the same time, they won't interfere with normal XML tags that might be used in code etc.

The text within the tags should be normal text tokens though, which makes them more flexible during deployment. This way, you could, for example, write `<author>GOD ALMIGHTY</author>` tags during inference and see some interesting behaviors. I will show more use-cases further down.

To be clear, I think we do need to train a bit without tags, or with some tags dropped, or the tags in random order, or tags with empty labels, etc., to make the models more robust. A lot of this will happen naturally (you will use different tags for blog posts than for books, or reasearch papers, or legal documents, etc.), but using such data-augmentations on purpose from time to time is likely a good idea.

## Why does this lead to a separation of simulacra?

A model can learn many things, but what it says depends on its conditioning. When the conditioning between different texts is very indistinct, the model will learn the most common completion. This is by necessity: even if it is possible for the model to memorize all the possible completions, producing the most common one will minimize loss if the LLM cannot successfully infer the context of the text. This is true whether memorization means actual memorization or the learning of generalized functions.

Just adding an author, date, and source tag will do wonders here, enabling the model to learn and, importantly, express much more distinct simulations, conditioned on much more explicit inputs. If a single source says the same thing over and over again in response to some prefix, the model can easily predict that completion given tags, reducing the loss and thus the weight of that source. If there is another source with a different tag, which completes the same prefix a different way, the model can learn to differentiate the two, and build a more complex and complete understanding of what's going on. With luck, this contextual understanding will generalize beyond the tags themselves, though I'm not certain that it will.

Here's an analogy: RL uses Chains of Thought as training data; RL simply weights the gradients of those CoTs by data quality. Tags allow the model to do essentially the same thing with any piece of text during pre-training.

The tags are similar to a system prompt; but because they apply throughout pre-training, and are so distinct from the other tokens, there is a real separation between the guidance that they give and the rest of the text. In other words, a soft but real separation between program and data can potentially be achieved via tags, because the tags impact the *program* simulated by the LLM so strongly that other inputs are more distinctly data-only. Obviously, this wouldn't fully fix everything, but I believe it could help.

## Examplary use-cases

This section gives examples of potential advantages brought about by this design

### Compartmentalize the Waluigi

An agent needs to know bad behaviors in order to express only good ones, so when we train a model to behave in extreme manners, like never telling a user how to make Meth, we also train it to simulate the character that will always explain how to make Meth: its [Waluigi](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post). When successfully prompted to deviate from this extreme behavior of refusals, it can surface its Waluigi, making it generally unsafe. And because a model will never be 100% successful at always refusing, this means that safety training will always introduce a hidden vulnerability.

Tags might help compartmentalize the two. If we train our desired agent on one set of tags, and its Waluigi on a different one, we might not be forced to train both behaviors&mdash;the wanted one and, implicitly, its opposite&mdash;into the same simulacra.

### Use-case specficity and character control

I'm not sure if the above will actually work. However, whether it does or not, using tags allows us much better character-control: 8kun, Wikipedia, arXiv, and X all have very different connotations, and will heavily influence the character. The same is true for the `<author>` tag, the `<date>` tag (though to a lesser degree), and others. And it is obvously easy to combine tags that don't appear together in pre-training (like `<url>https://x.com</url>` together with `<isbn>...</isbn>`), put multiple values into one tag (like `<author>God Almighty, Satan</author>`), or repeat tags.

That is obviously good for alignment.

If you're a company, you can get a head-start on your character-design simply by using the right tags in post-training, and if you're and individual like [@repligate](https://x.com/repligate) or [@anthrupad](https://x.com/anthrupad) who is into exploring LLM behavior deeply, you now have a tool for much more explicit illumination of the dusty corners of behavior-space.

> Note: model providers might not allow for access to those special tokens in fear of giving users too much control. While I'm not a fan of the attitude, I understand its business logic, and offer the following compromise: do offer users control over the tags, but police the tags the way you currently police the text outputs themselves. This would likely be easier, and still give your users access to powerful capabilities. Again, I don't find this super appealing, and Open Source competitors will gain an advantage by simply making the tags freely controllable, but it is a decent option.

But it's also a big win for capabilities.

The separation of simulacra might allow you to train multiple distinct agents for different use-cases, and bring forth the correct one simply by changing the tags: a chat agent, a research agent, a CLI agent, etc. This is great for distribution and user experience, but it goes further than that.

The separation of simulacra allows for specialist agents within one model; all expressing different behaviors, and bringing forth certain skills via RL, without diminishing the other agent. All would likely still have access to the full knowledge and skill set, and potentially even benefit from the others' training (though that's just a hope of mine). The different agents could even serve as judges or teachers for other agents expressed by the same LLM.

It also makes it easy to re-combine the agents: just write multiple ones into the tags, or actively mix the embeddings.

### Continual Learning

The ultimate capability that the separation of simulacra might enable is continual learning.

An issue with continual learning is catastrophic forgetting. Teach the model new facts and it will forget old ones. However, I believe that this is at least in part due to LLMs being [Contextualization Machines](https://stochasm.blog/posts/contextualization-machines/). They produce outputs *conditional on the context*. If the context does not leave enough criterial with which to differentiate it from another context, then of course following it up with text B will lead the model to unlearn to complete this context with text A, even if it was previously trained to produce text A from the context.

Tags work around this issue by providing additional context which allows the model to retain the ability to express previously learned information *in the context in which it learned it*.

Beside the obvious use-cases of continual learning, this might allow us to provide an agent with new information *in the weights* without overwriting its behavior, simply by training the information conditioned on different tags. This new information should be available to the agent. For example, we could train on a new paper, without the agent suddenly sounding like it wants to lecture you. In general, a model will contextualize new data better and thus be less likely to overwrite previous knowledge.

As a sidenote, the same effects might make curriculum learning more effective. A `<date>...</data>` tag will help the model contextualize new information much better, and thus help it make connections between different pieces of information. This is of course true for any of the tags, but it's especially relevant here.

## Why tags might be a bad idea

Current LLMs are trained in an environment that is typically devoid of explicit context. But as prediction success is highly dependent on knowing the context you're in, LLMs are forced to become extremely good at [inferring the Logos](https://x.com/jd_pressman/status/1918419540788297856), a.k.a. understanding the process that lead to the generation of the text. Who wrote it? Why? On what platform or medium? When?

Providing much of this context via tags might reduce the incentive for models to gain this intuition.

And while it'd make explicit control of the models easier, it's possible that it would also reduce the ways in which LLMs could be steered in more subtle manners. Some companies would prefer that: it's a security boon, and makes it easier for companies to guardrail the models without the need for schizophrenic LLM whisperers, and increase reliability from the perspective of customers. On the other hand, it is also a capability that many will miss for legitimate reasons.

The obvious solution to this dilemma is to leave out the tags completely for a decent percentage of the training data (10%? 20%?). This could actually improve the model's ability to infer context, because it has seen from what context different types of text are created, and if it guesses wrong, it will be updated strongly.

Another option is to add the tags at the end of the text in some samples. Ignore the loss at the special-token tags so that the model doesn't learn to just randomly produce them (or just actively set their probability to 0 during inference, or both), and actively train the model to infer context.

## Words of caution

I have to stress again that this is all just speculation.

It's possible that using tags would have all the advantages and disadvantages that I've discussed; it's possible that it only has the advantages, or only the disadvantages; it's possible that nothing in this article is correct, or that there would be almost no practical impact.

Additionally, I do not believe that any of the effects I've described would apply 100%; instead, tags would at best be a decent step in the right direction.

## Citation

```bibtex
@misc{snimu2025tags,
    title={\<Tags\> make them smart, \<Tags\> make them safe},
    author={Sebastian Nicolas Müller},
    year={2025},
    month={04},
    url={https://snimu.github.io/2025/05/03/tags-make-them-smart-tags-make-them-safe.html}
}
```
