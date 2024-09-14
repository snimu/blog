# Reflections on CrossMax

In [Ensemble everything everywhere: Multi-scale
aggregation for adversarial robustness](https://arxiv.org/abs/2408.05446), Stanislav Fort and Balaji Lakshminarayanan develop several techniques to make image-models (mainly CNNs, but also ViTs) robust against adversarial examples without adversarial training. One of their ideas is to use a CrossMax pooling operation to aggregate the outputs of multiple models, or multiple layers of the same model, into a single output.

CrossMax consits of two steps: 1.) substract the maximum class value per predictor, and 2.) substract the maximum prediction value per class. These steps must be applied in this order to work.

In this article, I will 1.) explain what CrossMax does and explain my view of why it increases adversarial robustness without reducing performance, and 2.) explain my view of why its two steps have to be applied in this order.

## What does CrossMax do and why does it increase adversarial robustness?

I will explain both steps on an extremely simplified example, shown as a table:

| &darr; predictor; class &rarr; | $c_1$ | $c_2$ | $c_3$ |
|---------------------------------|--------|--------|--------|
| $p_1$                           | ???    | ???    | ???    |
| $p_2$                           | ???    | ???    | ???    |
| $p_3$                           | ???    | ???    | ???    |

I have left out the values for the predictors for now, because I will fill them with whatever values best fit my example later.

Lets look at both steps in detail.

### Step 1: Substract the maximum class value per predictor

In a normal predictor, where this step isn't applied, an attacker can increase the probability of a class arbitrarily, make it the most likely class, and therefore succeed in the attack.

If we substract the maximum class value per predictor, we can make sure that the attacker can't increase the probability of a class arbitrarily. We effectively train the predictor to put the most likely class in the second most likely poisition. Therefore, an attacker must tune their attack such that it puts the probability of the desired class exaclty in the second most likely position. This is much more difficult than arbitrary class increases.

Here is an example of a set of predictors' outputs, before `softmax`:

| &darr; predictor; class &rarr; | $c_1$ | $c_2$ | $c_3$ |
|---------------------------------|--------|--------|--------|
| $p_1$                           | $1.2$   | $0.8$    | $0.3$    |
| $p_2$                           | $0.8$    | $0.9$    | $0.1$    |
| $p_3$                           | $1.1$    | $0.7$    | $0.5$    |

When you substract the maximum class value per predictor, you get:

| &darr; predictor; class &rarr; | $c_1$ | $c_2$ | $c_3$ |
|---------------------------------|--------|--------|--------|
| $p_1$                           | $0.0$   | $-0.4$    | $-0.5$    |
| $p_2$                           | -0.1$    | $0.0$    | $-0.8$    |
| $p_3$                           | $0.0$    | $-0.4$    | $-0.6$    |

What becomes clear quickly is that the distribution of the remaining classes&mdash;all classes but the most likely one&mdash;retains its order. 

In fact, when you apply `softmax` to both of these tensors, the results is the same! This explains why doing this first step of CrossMax alone changes nothing about the 


