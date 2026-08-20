# Minimal Character-Level RNN — From Scratch

A learning-oriented implementation and mathematical walkthrough of a **vanilla character-level Recurrent Neural Network (RNN)**.

This README explains the ideas behind Andrej Karpathy's well-known minimal NumPy character-level RNN and connects each part of the implementation to the underlying mathematics: forward propagation, softmax, cross-entropy loss, Backpropagation Through Time (BPTT), gradient clipping, and Adagrad.

## Attribution and content credit

This project is **educational and inspired by Andrej Karpathy's original `min-char-rnn.py` implementation**:

- Original author: **Andrej Karpathy**
- Original gist: https://gist.github.com/karpathy/d4dee566867f8291f086
- Original implementation: `min-char-rnn.py`
- The original gist identifies itself as a "Minimal character-level Vanilla RNN model" written by Andrej Karpathy and states that it is released under the **BSD License**.

The architecture, core equations, and training flow discussed here closely follow that implementation. This README is an explanatory/educational work and should not be interpreted as claiming authorship of Karpathy's original implementation.

Please refer to the original gist and its license before redistributing or modifying the original source code.

---

# 1. What are we building?

We are building a **character-level language model**.

Given a sequence of characters:

```text
hello
```

the model learns to predict the next character:

```text
hel -> l
ell -> o
```

More generally:

```text
previous characters
        ↓
      RNN
        ↓
probability distribution
        ↓
next character
```

During generation, the predicted character is fed back into the model:

```text
seed
 ↓
predict next character
 ↓
append character
 ↓
use it as the next input
 ↓
predict again
 ↓
...
```

This is an **autoregressive language model**.

---

# 2. High-level architecture

The model has three important transformations:

```text
Character
   ↓
One-hot vector x_t
   ↓
┌─────────────────────────────┐
│ Vanilla RNN                 │
│                             │
│ h_t = tanh(Wxh x_t          │
│            + Whh h_{t-1}    │
│            + b_h)           │
└─────────────────────────────┘
   ↓
Hidden state h_t
   ↓
y_t = Why h_t + b_y
   ↓
Softmax
   ↓
Probability of every character
```

The model maintains a hidden state:

```text
h_t
```

which acts as its memory of the preceding characters.

---

# 3. Data representation

Suppose the vocabulary is:

```text
['a', 'b', 'c', 'd']
```

We assign IDs:

```text
a -> 0
b -> 1
c -> 2
d -> 3
```

A character can then be represented using one-hot encoding:

```text
a -> [1, 0, 0, 0]
b -> [0, 1, 0, 0]
c -> [0, 0, 1, 0]
d -> [0, 0, 0, 1]
```

The original implementation creates this representation inside `lossFun()`:

```python
xs[t] = np.zeros((vocab_size, 1))
xs[t][inputs[t]] = 1
```

So if `inputs[t] == 2`, the vector becomes:

```text
[0, 0, 1, 0]^T
```

---

# 4. Training sequence

Suppose:

```text
hello
```

We can create:

```text
Input:   h e l l
Target:  e l l o
```

At each timestep:

```text
x_0 = h -> target e
x_1 = e -> target l
x_2 = l -> target l
x_3 = l -> target o
```

The target is therefore the input sequence shifted by one character.

Mathematically:

```text
x_t = current character
target_t = next character
```

The model is learning:

\[
P(x_{t+1}\mid x_0,\ldots,x_t)
\]

---

# 5. Model parameters

The original implementation defines:

```python
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
```

These are the trainable parameters.

## Wxh

Input-to-hidden weights:

\[
W_{xh}\in\mathbb{R}^{H\times V}
\]

where:

- \(H\) = hidden size
- \(V\) = vocabulary size

It maps:

```text
input vector → hidden representation
```

## Whh

Hidden-to-hidden/recurrent weights:

\[
W_{hh}\in\mathbb{R}^{H\times H}
\]

It determines how the previous hidden state influences the next hidden state.

## Why

Hidden-to-output weights:

\[
W_{hy}\in\mathbb{R}^{V\times H}
\]

It maps:

```text
hidden state → vocabulary logits
```

## bh

Hidden bias:

\[
b_h\in\mathbb{R}^{H\times1}
\]

## by

Output bias:

\[
b_y\in\mathbb{R}^{V\times1}
\]

---

# 6. Forward propagation

The heart of the vanilla RNN is:

\[
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
\]

This happens for every timestep.

## Step 1: input contribution

\[
W_{xh}x_t
\]

This asks:

> What does the current character contribute to the hidden state?

## Step 2: previous-memory contribution

\[
W_{hh}h_{t-1}
\]

This asks:

> What should I remember from the previous timestep?

## Step 3: combine them

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h
\]

## Step 4: non-linearity

\[
h_t=\tanh(a_t)
\]

The original code:

```python
hs[t] = np.tanh(
    np.dot(Wxh, xs[t])
    + np.dot(Whh, hs[t-1])
    + bh
)
```

---

# 7. Why tanh?

The RNN needs a non-linear activation function.

Without it:

\[
h_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h
\]

would be only a sequence of linear transformations.

`tanh` maps values into:

\[
[-1,1]
\]

and introduces non-linearity.

Its derivative is especially convenient:

\[
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
\]

That derivative appears directly during backpropagation.

---

# 8. Hidden state to output

After calculating the hidden state:

\[
h_t
\]

we calculate the raw output:

\[
y_t=W_{hy}h_t+b_y
\]

The original code calls these values:

```python
ys[t] = np.dot(Why, hs[t]) + by
```

These are called **logits**.

They are not probabilities yet.

---

# 9. Softmax

We convert logits into probabilities:

\[
p_{t,i}
=
\frac{e^{y_{t,i}}}
{\sum_j e^{y_{t,j}}}
\]

The original implementation does:

```python
ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
```

The result might look like:

```text
a -> 0.02
b -> 0.05
c -> 0.80
d -> 0.13
```

The probabilities sum to 1.

The highest probability represents the model's most likely next character.

---

# 10. Cross-entropy loss

Suppose the correct next character is:

```text
c
```

and the model predicts:

```text
a -> 0.02
b -> 0.05
c -> 0.80
d -> 0.13
```

The loss for this timestep is:

\[
L_t=-\log(p_{t,\text{correct}})
\]

Therefore:

\[
L_t=-\log(0.80)
\]

which is approximately:

\[
0.223
\]

If the model assigns a very small probability to the correct character:

\[
p=0.01
\]

then:

\[
-\log(0.01)\approx4.605
\]

So:

```text
High probability for correct answer
        ↓
     low loss

Low probability for correct answer
        ↓
    high loss
```

The original implementation:

```python
loss += -np.log(ps[t][targets[t], 0])
```

adds the loss for every timestep.

---

# 11. Sequence loss

For a sequence of \(T\) timesteps:

\[
L=\sum_{t=1}^{T}L_t
\]

Therefore:

\[
L=
-\sum_{t=1}^{T}
\log p(y_t\mid x_1,\ldots,x_t)
\]

The implementation accumulates:

```python
loss = 0

for t in ...:
    ...
    loss += -np.log(ps[t][targets[t], 0])
```

---

# 12. Why do we need backpropagation?

The forward pass tells us:

> How wrong was the model?

The loss gives us a number.

But we need to know:

> Which weights caused this error, and in which direction should they change?

That is what the gradients provide.

For each parameter \(\theta\), we want:

\[
\frac{\partial L}{\partial\theta}
\]

For example:

\[
\frac{\partial L}{\partial W_{xh}}
\]

tells us how much the loss changes when \(W_{xh}\) changes.

---

# 13. Backpropagation Through Time

An RNN reuses the same weights at every timestep:

```text
             Wxh       Wxh       Wxh
              ↓         ↓         ↓
x₁ ───────→ h₁ ─────→ h₂ ─────→ h₃
             ↑         ↑         ↑
             Whh       Whh       Whh
```

Because the same parameters are used repeatedly, an error at a later timestep can depend on earlier hidden states.

Therefore we must propagate gradients backward through the sequence.

This is called:

# Backpropagation Through Time (BPTT)

Conceptually:

```text
Forward:

x₁ → h₁ → h₂ → h₃ → output

Backward:

output ← h₃ ← h₂ ← h₁
```

---

# 14. The softmax + cross-entropy gradient

This is one of the most important equations in the implementation.

Let:

\[
p=\text{softmax}(y)
\]

and let the target be represented as a one-hot vector:

\[
t
\]

For softmax + cross-entropy:

\[
\boxed{
\frac{\partial L}{\partial y}=p-t
}
\]

This is why the implementation can simply do:

```python
dy = np.copy(ps[t])
dy[targets[t]] -= 1
```

Suppose:

```text
p = [0.1, 0.7, 0.2]
```

and the correct class is index 1:

```text
target = [0, 1, 0]
```

Then:

\[
dy=p-target
\]

so:

```text
[0.1, 0.7, 0.2]
-
[0,   1,   0]
=
[0.1, -0.3, 0.2]
```

That is the gradient with respect to the logits.

---

# 15. Gradient for Why

We have:

\[
y_t=W_{hy}h_t+b_y
\]

Therefore:

\[
\frac{\partial L_t}{\partial W_{hy}}
=
\frac{\partial L_t}{\partial y_t}
\frac{\partial y_t}{\partial W_{hy}}
\]

which gives:

\[
\boxed{
dW_{hy}=dy_t h_t^T
}
\]

The code:

```python
dWhy += np.dot(dy, hs[t].T)
```

We use `+=` because the same `Why` is used at every timestep, so gradients from every timestep must be accumulated.

---

# 16. Gradient for by

Because:

\[
y_t=W_{hy}h_t+b_y
\]

the derivative with respect to the bias is simply:

\[
\boxed{
db_y=dy_t
}
\]

The code:

```python
dby += dy
```

Again, gradients are accumulated over all timesteps.

---

# 17. Backpropagating into the hidden state

The hidden state affects the output.

We have:

\[
y_t=W_{hy}h_t+b_y
\]

Therefore:

\[
\frac{\partial L}{\partial h_t}
=
W_{hy}^Tdy_t
\]

But there is another path:

```text
h_t
 ↓
h_{t+1}
 ↓
h_{t+2}
 ↓
loss
```

So the hidden state also receives gradient from the **future timestep**.

The implementation combines both:

```python
dh = np.dot(Why.T, dy) + dhnext
```

Mathematically:

\[
\boxed{
dh_t=W_{hy}^Tdy_t+dh_{t+1}^{\text{recurrent}}
}
\]

This is the key idea behind BPTT.

---

# 18. Backpropagation through tanh

We have:

\[
h_t=\tanh(a_t)
\]

where:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h
\]

The derivative of tanh is:

\[
\frac{d\tanh(a)}{da}
=
1-\tanh^2(a)
\]

Since:

\[
h_t=\tanh(a_t)
\]

we can write:

\[
\boxed{
\frac{\partial h_t}{\partial a_t}
=
1-h_t^2
}
\]

Therefore:

\[
dhraw
=
(1-h_t^2)\odot dh
\]

where \(\odot\) means element-wise multiplication.

The original code:

```python
dhraw = (1 - hs[t] * hs[t]) * dh
```

---

# 19. Gradient for Wxh

We have:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h
\]

Therefore:

\[
\boxed{
dW_{xh}=dhraw_t x_t^T
}
\]

The implementation:

```python
dWxh += np.dot(dhraw, xs[t].T)
```

Again, we accumulate gradients across all timesteps.

---

# 20. Gradient for Whh

Because:

\[
a_t=W_{hh}h_{t-1}+...
\]

we get:

\[
\boxed{
dW_{hh}=dhraw_t h_{t-1}^T
}
\]

The implementation:

```python
dWhh += np.dot(dhraw, hs[t-1].T)
```

This is especially important because `Whh` is responsible for carrying information from one timestep to another.

---

# 21. Gradient flowing to the previous timestep

The current hidden state depends on:

\[
h_{t-1}
\]

through:

\[
W_{hh}h_{t-1}
\]

Therefore:

\[
\boxed{
dhnext=W_{hh}^Tdhraw
}
\]

The code:

```python
dhnext = np.dot(Whh.T, dhraw)
```

Then, when the loop moves to:

```text
t-1
```

that gradient is included:

```python
dh = np.dot(Why.T, dy) + dhnext
```

This is literally the mechanism of Backpropagation Through Time.

---

# 22. Complete backward flow

At timestep \(t\):

```text
loss
 ↓
softmax
 ↓
dy = p - target
 ↓
┌──────────────────────────┐
│                          │
↓                          ↓
dWhy                      dh
                           ↓
                       through tanh
                           ↓
                         dhraw
                       /    |    \
                      /     |     \
                  dWxh    dWhh     dbh
                            ↓
                          dhnext
                            ↓
                       previous step
```

And this repeats backwards:

```text
t = T
 ↓
T-1
 ↓
T-2
 ↓
...
0
```

---

# 23. Why exploding gradients happen

The recurrent gradient repeatedly contains terms involving:

\[
W_{hh}^T
\]

and the derivative of tanh:

\[
1-h_t^2
\]

Very roughly, repeated multiplication can behave like:

\[
W_{hh}^T
W_{hh}^T
W_{hh}^T
\cdots
\]

If the effective magnitude is greater than 1, gradients can grow exponentially.

For example:

\[
2^{10}=1024
\]

\[
2^{20}=1,048,576
\]

This is the **exploding gradient problem**.

---

# 24. Gradient clipping

Karpathy's implementation clips every gradient:

```python
for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam)
```

Conceptually:

\[
g\leftarrow\text{clip}(g,-5,5)
\]

So:

```text
gradient = 12
       ↓
gradient = 5
```

and:

```text
gradient = -9
       ↓
gradient = -5
```

This prevents extremely large gradients from causing destructive parameter updates.

---

# 25. Vanishing gradients

The opposite problem also exists.

Since:

\[
\tanh'(x)=1-\tanh^2(x)
\]

the derivative is at most 1 and often substantially smaller.

Repeated multiplication can therefore produce:

\[
0.5^{10}=0.0009765625
\]

and:

\[
0.5^{20}\approx9.54\times10^{-7}
\]

So information from very early timesteps can become almost irrelevant.

This is the **vanishing gradient problem**.

This limitation is one of the major motivations for architectures such as **LSTM** and **GRU**.

---

# 26. Parameter update

Once we have the gradients, we need to update the parameters.

The simplest gradient descent equation is:

\[
\theta\leftarrow\theta-\eta\nabla_\theta L
\]

where:

- \(\theta\) = parameter
- \(\eta\) = learning rate
- \(\nabla_\theta L\) = gradient

Karpathy's implementation uses **Adagrad**, not plain gradient descent.

---

# 27. Adagrad

For every parameter, Adagrad keeps an accumulated squared-gradient memory:

\[
m_t=m_{t-1}+g_t^2
\]

Then:

\[
\boxed{
\theta_t
=
\theta_{t-1}
-
\frac{\eta g_t}
{\sqrt{m_t+\epsilon}}
}
\]

The implementation:

```python
mem += dparam * dparam

param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
```

The effect is that parameters with consistently large gradients receive progressively smaller effective learning rates.

---

# 28. Why initialize weights with small random values?

The implementation uses:

```python
np.random.randn(...) * 0.01
```

This gives small random initial values.

If all weights were exactly zero, many neurons would behave identically and symmetry would not be broken effectively.

Small random initialization allows different hidden units to learn different representations.

---

# 29. Why reset hprev?

The code contains:

```python
hprev = np.zeros((hidden_size,1))
```

This resets the RNN's memory.

When starting a new independent sequence, we don't necessarily want hidden state from an unrelated part of the dataset.

Conceptually:

```text
previous sequence
      ↓
   reset h
      ↓
new sequence
```

The original code also carries `hprev` between contiguous chunks when appropriate, which is a form of stateful training across consecutive text chunks.

---

# 30. Text generation

The `sample()` function performs generation.

It starts with a seed character:

```python
x[seed_ix] = 1
```

Then repeatedly:

\[
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
\]

\[
y_t=W_{hy}h_t+b_y
\]

\[
p_t=\text{softmax}(y_t)
\]

Then it samples:

```python
ix = np.random.choice(
    range(vocab_size),
    p=p.ravel()
)
```

Notice something important:

**It samples from the probability distribution rather than always taking `argmax`.**

Therefore generation can be stochastic.

---

# 31. Training vs generation

## Training

```text
Input characters
       ↓
Forward pass
       ↓
Predicted probabilities
       ↓
Cross-entropy loss
       ↓
BPTT
       ↓
Gradients
       ↓
Gradient clipping
       ↓
Adagrad
       ↓
Updated weights
```

## Generation

```text
Seed character
       ↓
RNN
       ↓
Softmax
       ↓
Probability distribution
       ↓
Sample character
       ↓
Feed sampled character back
       ↓
Repeat
```

---

# 32. Why the model can generate text

After training, the model has learned parameters that approximate:

\[
P(c_{t+1}\mid c_1,\ldots,c_t)
\]

For example, given:

```text
"th"
```

it might learn:

```text
e -> 0.80
a -> 0.05
i -> 0.04
o -> 0.03
...
```

Sampling from that distribution gives the next character.

Then the new context becomes:

```text
"the"
```

and the model predicts again.

That creates a chain:

```text
"t"
 ↓
"h"
 ↓
"e"
 ↓
" "
 ↓
"c"
 ↓
"o"
 ↓
"m"
 ↓
...
```

---

# 33. The complete mathematical pipeline

At each timestep:

### Input

\[
x_t
\]

### Hidden state

\[
\boxed{
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
}
\]

### Logits

\[
\boxed{
y_t=W_{hy}h_t+b_y
}
\]

### Probabilities

\[
\boxed{
p_t=\text{softmax}(y_t)
}
\]

### Loss

\[
\boxed{
L_t=-\log p_t[target_t]
}
\]

### Sequence loss

\[
\boxed{
L=\sum_tL_t
}
\]

### Output gradient

\[
\boxed{
dy_t=p_t-target_t
}
\]

### Output weights

\[
\boxed{
dW_{hy}+=dy_t h_t^T
}
\]

### Hidden gradient

\[
\boxed{
dh_t=W_{hy}^Tdy_t+dhnext
}
\]

### Through tanh

\[
\boxed{
dhraw_t=(1-h_t^2)\odot dh_t
}
\]

### Input weights

\[
\boxed{
dW_{xh}+=dhraw_t x_t^T
}
\]

### Recurrent weights

\[
\boxed{
dW_{hh}+=dhraw_t h_{t-1}^T
}
\]

### Previous hidden state

\[
\boxed{
dhnext=W_{hh}^Tdhraw_t
}
\]

### Update

\[
\boxed{
\theta\leftarrow\theta-\text{AdagradUpdate}(\nabla_\theta L)
}
\]

---

# 34. The most important equations to memorize

If you're learning RNNs, don't try to memorize every line of the implementation immediately.

Understand these five:

### 1. RNN state

\[
\boxed{
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
}
\]

### 2. Output

\[
\boxed{
y_t=W_{hy}h_t+b_y
}
\]

### 3. Softmax

\[
\boxed{
p_i=\frac{e^{y_i}}{\sum_j e^{y_j}}
}
\]

### 4. Cross entropy

\[
\boxed{
L_t=-\log(p_{target})
}
\]

### 5. Softmax + cross-entropy gradient

\[
\boxed{
\frac{\partial L}{\partial y}=p-target
}
\]

Once these are clear, the rest of Karpathy's implementation becomes much easier to follow.

---

# 35. Mapping the math directly to the code

| Mathematical operation | Code |
|---|---|
| One-hot \(x_t\) | `xs[t][inputs[t]] = 1` |
| Hidden state | `hs[t] = np.tanh(...)` |
| Logits | `ys[t] = np.dot(Why, hs[t]) + by` |
| Softmax | `ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))` |
| Cross entropy | `loss += -np.log(ps[t][targets[t],0])` |
| Softmax gradient | `dy[targets[t]] -= 1` |
| \(dW_{hy}\) | `dWhy += np.dot(dy, hs[t].T)` |
| \(db_y\) | `dby += dy` |
| Hidden gradient | `dh = np.dot(Why.T, dy) + dhnext` |
| Tanh derivative | `dhraw = (1-hs[t]*hs[t])*dh` |
| \(dW_{xh}\) | `dWxh += np.dot(dhraw, xs[t].T)` |
| \(dW_{hh}\) | `dWhh += np.dot(dhraw, hs[t-1].T)` |
| Previous-state gradient | `dhnext = np.dot(Whh.T, dhraw)` |
| Gradient clipping | `np.clip(dparam, -5, 5, out=dparam)` |
| Adagrad | `mem += dparam*dparam` |
| Parameter update | `param += -learning_rate*dparam/np.sqrt(mem+1e-8)` |

---

# 36. Important historical/context note

Karpathy's gist is intentionally **minimal**: it implements the RNN directly with NumPy rather than relying on a framework's automatic differentiation or recurrent layer abstraction. The gist describes itself as a minimal character-level vanilla RNN and includes the forward pass, BPTT, gradient clipping, sampling, and Adagrad training loop in a compact implementation. citeturn0search0turn0view0

That makes it particularly useful for learning **what TensorFlow/PyTorch are doing underneath the abstractions**.

Your earlier TensorFlow code:

```python
SimpleRNN(...)
```

hides most of this machinery.

Karpathy's implementation exposes it:

```text
TensorFlow SimpleRNN
       ↓
    abstraction

Karpathy NumPy RNN
       ↓
Wxh
Whh
Why
bh
by
forward pass
loss
BPTT
gradients
clipping
Adagrad
sampling
```

So studying this implementation is an excellent next step before moving to LSTM.

---

# 37. Suggested learning progression

I would study the implementation in this exact order:

```text
1. Character vocabulary
        ↓
2. One-hot encoding
        ↓
3. Wxh / Whh / Why
        ↓
4. Forward pass
        ↓
5. Hidden state
        ↓
6. Softmax
        ↓
7. Cross-entropy loss
        ↓
8. dy = p - target
        ↓
9. Backpropagation
        ↓
10. BPTT
        ↓
11. Vanishing/exploding gradients
        ↓
12. Gradient clipping
        ↓
13. Adagrad
        ↓
14. Sampling
```

Once these are comfortable, the next natural step is:

```text
Vanilla RNN
     ↓
Why gradients vanish
     ↓
LSTM
     ↓
Cell state
     ↓
Forget gate
     ↓
Input gate
     ↓
Output gate
     ↓
LSTM backpropagation
```

That will make LSTM much easier to understand than treating it as just another Keras layer.

---

## Original reference

urlAndrej Karpathy — Minimal character-level Vanilla RNN gisthttps://gist.github.com/karpathy/d4dee566867f8291f086

The original gist is the source being studied and credited above. It currently identifies the file as `min-char-rnn.py`, authored by Andrej Karpathy, and labels it under the BSD License. citeturn0view0
