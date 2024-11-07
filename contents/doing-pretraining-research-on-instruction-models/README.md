# Doing Pre-training Research on Instruction Models

I want to do research on pre-training methods, which is of course best done on base models. However, there is a conflict: evals are often better done on instruction models. Here is a simple way to resolve this conflict.

We just need to have a model that is availe both as a base model and as an instruction model; for example, [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

## Method

First, we take the difference between the instruction and base model:

$$
\Delta \Theta = \Theta_{\mathrm{instruct}} - \Theta_{\mathrm{base}}
$$

Then, we train the base model on our data, with whatever methods we want to research (as defined by the $\mathrm{\texttt{train}}$ function):

$$
\Theta_{\mathrm{trained}} = \mathrm{\texttt{train}}(\Theta_{\mathrm{base}})
$$

Finally, we re-add $\Delta \Theta$ to the resulting weights:

$$
\Theta_{\mathrm{final}} = \Theta_{\mathrm{trained}} + \Delta \Theta
$$

If the norm of $\Theta_{\mathrm{trained}} - \Theta_{\mathrm{base}}$ is small compared to that of $\Delta \Theta$, then I expect this to work.

## Additional information

So what is $\Delta \Theta$ for the Llama-3.2-1B models?

I ran the following code (and put the results in the final comment / string):

```python
"""
INSTRUCTIONS:

- you need to run `huggingface-cli login` once and add your token
- you need access to the meta-llama weights

ON LAMBDA LABS:

- you need to run `pip install -U torch torchvision` before running this script
"""
import argparse
from typing import Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM

def load_model(which: Literal["base", "instruct"] = "base") -> nn.Module:
    if which == "base":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    return model


def measure_weight_norm(model: nn.Module, measure: Literal["abs", "square"] = "square") -> float:
    numels = []
    norms = []

    for param in model.parameters():
        numels.append(param.numel())
        norms.append(torch.abs(param).sum().item() if measure == "abs" else param.pow(2).sum().item())

    return sum(norms) / sum(numels)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="square", choices=["abs", "square"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    base_model = load_model("base")
    instruct_model = load_model("instruct")

    base_norm = measure_weight_norm(base_model, measure=args.measure)
    instruct_norm = measure_weight_norm(instruct_model, measure=args.measure)

    for param_base, param_instruct in zip(base_model.parameters(), instruct_model.parameters()):
        param_instruct.data = param_instruct.data - param_base.data

    diff_norm = measure_weight_norm(instruct_model, measure=args.measure)

    print(f"MEASURE: {args.measure}")
    print(f"Base model weight norm: {base_norm:.4f}")
    print(f"Instruct model weight norm: {instruct_norm:.4f}")
    print(f"Diff model weight norm: {diff_norm:.4f}")
    print(f"base_norm / instruct_norm: {base_norm / instruct_norm:.4f}")
    print(f"diff_norm / instruct_norm: {diff_norm / instruct_norm:.4f}")
    print(f"diff_norm / base_norm: {diff_norm / base_norm:.4f}")

    """
    RESULTS:

    MEASURE: abs
    Base model weight norm: 0.0154
    Instruct model weight norm: 0.0150
    Diff model weight norm: 0.0026
    base_norm / instruct_norm: 1.0260
    diff_norm / instruct_norm: 0.1744
    diff_norm / base_norm: 0.1699

    MEASURE: square
    Base model weight norm: 0.0004
    Instruct model weight norm: 0.0004
    Diff model weight norm: 0.0000
    base_norm / instruct_norm: 1.0532
    diff_norm / instruct_norm: 0.0285
    diff_norm / base_norm: 0.0270
    """
```

A $17\%$ difference in absolute weight norms between $\Delta \Theta$ and $\Theta_{\mathrm{base}}$ is a good sign, because it leaves plenty of room for changing the base model weights through your training, and still having the instruct-weights be meaningful.

## Caveats

- I haven't tested this
- So it might not work
- I wanted to make the idea public and will hopefully try it soon on something I've been cooking for a while

## Citation

If you use this, please cite:

```latex
@misc{muller2024pretraining,
    title={Doing Pre-training Research on Instruction Models},
    author={Sebastian M\"uller},
    year={2024},
    month={nov},
    url={https://github.com/snimu/blog/blob/main/contents/doing-pretraining-research-on-instruction-models/README.md}
}
```
