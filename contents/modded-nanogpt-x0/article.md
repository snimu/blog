# modded-nanogpt: Embeddings Extravaganza 2 (x00-x04)

modded-nanogpt (LINK) adds the token-embedding x0 to the residual stream in a weighted sum before every layer. What happens if we create additional embeddings layers, and add more and more of them to the weighted sum?

## The setup

... TODO (explain baseline and what's added and how, plus initilization; will call each run by the maximum new embedding, so x01 for one added embedding, etc.; mention my prior work on looking at the lambdas)

## Validation losses

Let's first look at the validation losses per step:

![Validation losses over steps](images/val_loss-step.png)

As you can see, you can see nothing. We need to zoom in further to the end of training:

![Validation losses over steps 5500-6000](images/val_loss-step-5500-6000.png)

The obvious pattern that emerges is that with each additional embedding layer, the model learns more in each step. However, modded-nanogpt is all about performance per time, so let's plot that as well. This time, we'll simply look at the zoomed in last few steps:

![Validation losses over time, seconds 1400-1550](images/val_loss-time-1400-1550.png)

What immediately jumps out is that beyond two additional embeddings, the performance per time-step degrades quickly. And even before that, the baseline is clearly the best throughout the majority of training. However, toward the very end, two runs with added embeddings seem to actually be better: x01 and x02. The differences are tiny though, and I chalk them up to random chance.

There is a complication to this, though: torch compile flags. In all of the above runs (including the baseline), I removed the `torch._dynamo.config.compiled_autograd = True` flag, because it cause an error in flexattention that I didn't want to deal with; and I removed the `_patched_trace_structured` function because it cause some other error that I didn't want to deal with. I have, however, kept the `torch._inductor.config.coordinate_descent_tuning = True` flag. In [the latest record log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/042225_GPT2Medium_Record8/075_640429f2-e726-4e83-aa27-684626239ffc.txt), there is a comment behind it that says `# we have banned this flag for new records because it causes compilation to take 30min` (though this comment isn't present in `train_gpt_medium.py`, and I measure around 4 minutes of compile and warmup time). So I ran the baseline and the first two experiments again without the flag, 5 times each. For each step, I averaged the time over the 5 runs, and the loss; both independently. That's not quite correct, but it gives a pretty good feel for what's going on. Here are the resulting loss curves:

![Val Losses without coordinate_descent_tuning](images/val_loss_time_record.png)

In this case, adding a single additional embedding causes a very clear record.

Below, I will try to determine how the model uses the additional embedding to improve its per-step (and, depending on the flag, per-time-step) performance.

## Lambdas

To me, the most interesting part about these results is that adding more embedding layers just keep improving model performance per step. In some sense, that's expected, because embedding layers are just cheap additional parameters. On the other hand, I do find it somewhat surprising: these embeddings look at each token individually, so on their own they can at best achieve 1-gram accuracy, which is bad. They must help the model learn to make better use of the main layers somehow; let's look at the lambdas in order to find out how.

### x01

My expectation was that x00 will be high in the beginning and low in the end; and that x01 will be low in the beginning and high in the end. I expected this to be a continuous change. The reason is that I expected x00 to contain training statistics about the input embeddings, a clean representation of what the tokens mean, while x01 contains 1-gram statistics about the most likely next token-distribution given a single token. Then, x00 would become less useful over the layers and x01 more useful (though I didn't expect this separation of concerns to be nearly this clean or one-dimensional).

Here's how the lambdas for x, x00, and x01 acutally look at the end of training for the different layers, averaged over 5 runs (because they are a bit noisy):

![x00, x01: mean over 5 runs](images/0-layer-mean5.png)

There are so many things to say!

- As always, layer 8 is a bit of an exception to everything I'm going to say below, but that's just because there is no attention at layer 7, which influences all lambdas in layer 8 significantly
- At the input, x == x00, and both are very high, suppressing x01
- Afterward, the weights of x falls to a constant low value over the layer, and it barely changes until the very end
- The weight of x00 is noisy, but seems to fall pretty consistently from layer to layer
- The weight of x01 is a very low positive value throughout the layers
- That's until the very end&mdash;layers 12-15&mdash;where x00 suddenly drops like a stone into negative territory, and x01 rises quickly

What could explain this behavior? Here's my speculation. Let me be clear: it's nothing *but* speculation, and even if it's true it won't be 100% crisply be true because that's almost never the case in Deep Learning. Nevertheless, here I go:

Assume that x00 is a pure representation of the input tokens; and that x01 is a 1-gram embedding of the prior probability distribution over the most likely next token, for each token in the vocabulary. Then, the following pattern emerges:

- In the last few layers (12-15):
  - x01 explodes in weight, providing a prior for the next-token prediction
  - x is used to adjust this next-token prediction in a more data-dependent manner; apparently, not much is needed for this task
  - x00 is substracted strongly from the residual stream, which undoes how strongly it was added to it in the earlier layers, leaving a clean next-token prior from x01 and adjustment from x
- In the earlier layers (1-11):
  - x00 dominates the forward pass, followed in weight by x, while x01 is very low
  - This allows the model to very slowly develop the adjustment direction x over the layers, given mostly the view of the input tokens x00, and, to a lower degree, the previous attempt at creating the adjustment (and, to an even lower degree, the next-token prior)
- In layer 0:
  - x == x00, and both dominate strongly
  - This suggests that the model just wants to initialize the adjustment vector very strongly; and that in turn supports the idea that this adjustment vector will simply be carefully refined over the subsequent layers

One potential upset to this theory is that x00 and x01 are RMS-normed, but x isn't (except before layer 0 and after layer 15). Of course, x is always normed at the input to the Attention and MLP layers, but not at their outputs, and thus never at the residual, where the weighted sum between x, x00, and x01 always occurs. This means that it's possible that the norm of x could be rising monotonically, or vary wildly.

I don't believe that that's the case though; if it were, SGD would likely vary both the norm of x *and* the x-lambda in the same direction, at least somewhat. I would therefore assume that the norm of x is fairly stable throughout. In fact, I can *imagine* that x00 (and x01) stabilize the norm of x. The hypothesis is: x00 and x01 are normed, while x floats freely, so it's easier to control the relative weights of all three by varying the lambdas of x00 and x01, while keeping the both the lambda and the norm of x fairly constant, than to control both the lambda and norm of x. I'm very unsure about this though.

---

So I measured some outputs to test the above predictions!

#### Predictions

Firstly, and most importantly, I took 32 input tokens, ran a forward pass over them, and saved the top-10 predictions for x (at the input of the last layer, before it'a mixed with x00, x01, or the value embeddings), x00, x01, and the value-embeddings ve0, ve1, and ve2. You can find the full ten predictions under [predictions.md](https://github.com/snimu/modded-nanogpt-experiments/blob/main/experiments/00004-x0/predictions.md), together with the predicted probability for each token, and the code that produced the results [at this file](https://github.com/snimu/modded-nanogpt-experiments/blob/main/experiments/00004-x0/runs/4-2025-08-22-x00-x01-with-word-logging.py). Below, I simply show the most likely next token at each position, for each of the different components.

The dictionary shows one sub-dictionary for each component (x, x00, etc.). Each of these subdictionaries maps from the token position to another sub-sub-dictionary, which in turn maps from the true input tokens at each position to the most likely next token as decoded from each component by the language head.

The full input sentence is `mail account: email@example.com\n- Emails will include links. Nothing fancy. We??re keeping it simple.\n- We?`. The questionmarks are where my terminal didn't properly render the output bytes which I noticed too late. I belive it's just the "'" sign.

Here are the results:

```python
{
    'x': {
        0: {'mail': 'mail'},
        1: {' account': ','},
        2: {':': '\n'},
        3: {' email': '@'},
        4: {'@': 'example'},
        5: {'example': '.'},
        6: {'.': 'com'},
        7: {'com': '\n'},
        8: {'\n': '-'},
        9: {'-': ' Email'},
        10: {' Emails': ':'},
        11: {' will': ' be'},
        12: {' include': ' a'},
        13: {' text': ' and'},
        14: {' and': ' a'},
        15: {' links': ' to'},
        16: {'.': '\n'},
        17: {' Nothing': ' will'},
        18: {' fancy': '.'},
        19: {'.': '\n'},
        20: {' We': ' will'},
        21: {'�': '�'},
        22: {'�': 'll'},
        23: {'re': ' just'},
        24: {' keeping': ' it'},
        25: {' it': ' simple'},
        26: {' simple': '.'},
        27: {'.': '\n'},
        28: {'\n': '-'},
        29: {'-': ' We'},
        30: {' We': '�'},
        31: {'�': '�'}
    },
    'x00': {
        0: {'mail': ' address'},
        1: {' account': 'ancy'},
        2: {':': '\n'},
        3: {' email': ' address'},
        4: {'@': '#$'},
        5: {'example': ':'},
        6: {'.': '\n'},
        7: {'com': 'mission'},
        8: {'\n': 'The'},
        9: {'-': 'based'},
        10: {' Emails': ' are'},
        11: {' will': ' be'},
        12: {' include': ' a'},
        13: {' text': 'ured'},
        14: {' and': ' other'},
        15: {' links': ' between'},
        16: {'.': '\n'},
        17: {' Nothing': ' else'},
        18: {' fancy': 'pants'},
        19: {'.': '\n'},
        20: {' We': "'ve"},
        21: {'�': '�'},
        22: {'�': 's'},
        23: {'re': ' supposed'},
        24: {' keeping': ' pace'},
        25: {' it': 'alian'},
        26: {' simple': ' enough'},
        27: {'.': '\n'},
        28: {'\n': 'The'},
        29: {'-': 'based'},
        30: {' We': "'ve"},
        31: {'�': '�'}
    },
    'x01': {
        0: {'mail': 'historic'},
        1: {' account': ' account'},
        2: {':': ':'},
        3: {' email': 'emouth'},
        4: {'@': '@'},
        5: {'example': 'roph'},
        6: {'.': '.'},
        7: {'com': 'hip'},
        8: {'\n': '\n'},
        9: {'-': '-'},
        10: {' Emails': '@@@@'},
        11: {' will': ' will'},
        12: {' include': 'icide'},
        13: {' text': 'duction'},
        14: {' and': ' and'},
        15: {' links': 'abilia'},
        16: {'.': '.'},
        17: {' Nothing': 'fact'},
        18: {' fancy': ' overdose'},
        19: {'.': '.'},
        20: {' We': 'cia'},
        21: {'�': '�'},
        22: {'�': '�'},
        23: {'re': 's'},
        24: {' keeping': 'eded'},
        25: {' it': ' it'},
        26: {' simple': 'overed'},
        27: {'.': '.'},
        28: {'\n': '\n'},
        29: {'-': '-'},
        30: {' We': 'cia'},
        31: {'�': '�'}
    },
    've0': {
        0: {'mail': ' Triangle'},
        1: {' account': 'RL'},
        2: {':': 'fulness'},
        3: {' email': 'caster'},
        4: {'@': 'stone'},
        5: {'example': 'irs'},
        6: {'.': ' wherein'},
        7: {'com': 'ones'},
        8: {'\n': 'zzo'},
        9: {'-': ' lat'},
        10: {' Emails': 'alls'},
        11: {' will': 'fork'},
        12: {' include': '.\'"'},
        13: {' text': 'hips'},
        14: {' and': 'elman'},
        15: {' links': 'owl'},
        16: {'.': ' wherein'},
        17: {' Nothing': ' firmware'},
        18: {' fancy': 'chall'},
        19: {'.': ' wherein'},
        20: {' We': 'amo'},
        21: {'�': 'aston'},
        22: {'�': 'olate'},
        23: {'re': 'ches'},
        24: {' keeping': 'ners'},
        25: {' it': 'OST'},
        26: {' simple': 'aling'},
        27: {'.': ' wherein'},
        28: {'\n': 'zzo'},
        29: {'-': ' lat'},
        30: {' We': 'amo'},
        31: {'�': 'aston'}
    },
    've1': {
        0: {'mail': 'ucl'},
        1: {' account': ' false'},
        2: {':': 'hon'},
        3: {' email': 'monary'},
        4: {'@': 'etz'},
        5: {'example': 'imate'},
        6: {'.': 'bol'},
        7: {'com': ' Caption'},
        8: {'\n': 'row'},
        9: {'-': 'imal'},
        10: {' Emails': 'ted'},
        11: {' will': 'RET'},
        12: {' include': 'hood'},
        13: {' text': 'sonian'},
        14: {' and': 'dom'},
        15: {' links': 'eric'},
        16: {'.': 'bol'},
        17: {' Nothing': 'ucket'},
        18: {' fancy': ')|'},
        19: {'.': 'bol'},
        20: {' We': 'tered'},
        21: {'�': ' naming'},
        22: {'�': ' files'},
        23: {'re': ' boss'},
        24: {' keeping': 'naires'},
        25: {' it': 'rd'},
        26: {' simple': 'ason'},
        27: {'.': 'bol'},
        28: {'\n': 'row'},
        29: {'-': 'imal'},
        30: {' We': 'tered'},
        31: {'�': ' naming'}
    },
    've2': {
        0: {'mail': 'heres'},
        1: {' account': 'went'},
        2: {':': ' �'},
        3: {' email': 'utes'},
        4: {'@': 'aqu'},
        5: {'example': 'Ms'},
        6: {'.': 'nikov'},
        7: {'com': 'ttes'},
        8: {'\n': 'ELF'},
        9: {'-': 'axis'},
        10: {' Emails': 'ift'},
        11: {' will': ' apologise'},
        12: {' include': 'baby'},
        13: {' text': ' length'},
        14: {' and': ' herself'},
        15: {' links': 'tering'},
        16: {'.': 'nikov'},
        17: {' Nothing': 'ened'},
        18: {' fancy': 'shire'},
        19: {'.': 'nikov'},
        20: {' We': 'gery'},
        21: {'�': '[/'},
        22: {'�': 'pered'},
        23: {'re': '?"'},
        24: {' keeping': ' Synt'},
        25: {' it': ' FTA'},
        26: {' simple': ' hen'},
        27: {'.': 'nikov'},
        28: {'\n': 'ELF'},
        29: {'-': 'axis'},
        30: {' We': 'gery'},
        31: {'�': '[/'}
    }
}
```

The first interesting result I'm seeing is for x00: while it's just the input embeddings, the language head interprets them as next-token predictions! "mail" &rarr; "address", "account" &rarr; "ancy", etc. are all clear 1-gram next-token predictions. I was right in my assumption that that's included, but wrong about which component represents it.

x01 sometimes contains a copy of the input, sometimes a seemingly random token. However, the top-10 predictions who that they often include similarly-sounding or -written words, or variants. For example, the next prediction for " account" is `[' account', 'agons', 'berry', ' Account', 'teenth', 'account', ' accounts', 'oku', 'colored']`. That's somewhat random, but some variant of "account" appears very often.

So my first intermediate conclusion is that the language head treats the input embeddings of current tokens as embeddings of the next token for each, and x01 makes up for that by providing (very approximate) information about the input embeddings. I believe that this switch is caused by the value embeddings: the same value embeddings are applied to layers 0 and 13, and 1 and 14, and 2 and 15. So the same embeddings must perform some task early and late in the model, which means that it's likely valuable for the model to interpret them differently at different points of the model. This has the nice side-effect that it minimizes the amount of change to x required over the length of the model, because leaving it un-changed makes it simply the 1-gram next-token predictions.

The next observation is that x also makes next-token predictions. So if both x and x00 are next-token predictions, why is x00 removed from the residual at the last few layers? One possible adaptation to my hypothesis is this (guaranteed to be overly simplified again, and pretty likely to still be wrong):

- x00 provides a 1-gram next-token prediction distribution at every layer of the model
- In early layers, this helps the model pick a direction for x, by biasing it
- In the last few layers, this prediction is substracted from x, to reduce the 1-gram bias which is of course sub-optimal, leaving only the refined x
- x01 gives an approximate view of the input, and the contrast between it and the next-token predictions by x00 gives the model additional information about how exactly it needs to refine the vector to make the best possible predictions

As for the value embeddings, they are (1) very different, and (2) make no sense. The latter isn't surprising; after all, they are applied inside the attention mechanism, not to the residual directly, so they are two transformations away from where the language head is usually applied. I just thought it would have been really interesting if they did make sense.

#### Norms

Typically, the norm of the activations rises from layer to layer if we don't use post-norm, which we don't. Since the embeddings x00 and x01 *don't* change in norm, this changes the total weight of each component significantly.

So I measured the norms of x over the different layers to make up for this phenomenon.

Unfortunately, I used `torch.linalg.norm` to measure the norm, while I used RMS-norm on to normalize the embeddings during the forward pass; so this doesn't fit perfectly. As I didn't think of this, I didn't measure the norms of x00 and x01; but x00 is x at the first layer, so the first norm of x is also the norm of x00, and I simply assumed the same for x01. So I divided all norms of x over the layers by the norm at the input of the first layer. Then, I multiplied the x lambda with the activation's norm at every layer. Finally, I normalized again by dividing each lambda at each layer by the sum over the absolute values of all lambdas at that layer. This way, we see a percentage weight of each component over the layers, taking the norm of the values into account (in an imperfect manner, unfortunately).

Here is that plot:

![Norm-corrected lambdas](images/0-layer-mean5-normalized-and-xnorm.png)

I can see two noteworthy phenomena.

The first is that the effects discussed above are very strongly visible even in the normalized plot, which is good to know.

The second is that in layer 8, the weight of x rises sharply, and that of x00 drops off a cliff. I assume that that's due to layer 7 not having an attention block, only an MLP, and since an MLP just approximates a dynamic embeddings layer for mixed tokens, it will be able to replace x00 at this point, but with more contextualized and thus more valuable information (because there were attention layers before it). I would guess that this makes x00 a distraction compared to what the MLP already adds to the residual stream, necessitating it to be set to approximately zero.
