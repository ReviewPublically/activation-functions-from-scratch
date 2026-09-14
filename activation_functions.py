"""
Activation Functions From Scratch — Built With NumPy
=======================================================
No PyTorch, no TensorFlow. Every activation function and its
derivative implemented and evaluated by hand, with real numbers,
not just described qualitatively.

Three things most explainers describe but never show:
  1. How much sigmoid and tanh gradients actually shrink at the
     extremes (the mechanical cause of vanishing gradients).
  2. A dying ReLU neuron caught in the act: a real weight update
     sequence where a neuron gets stuck outputting zero forever.
  3. Modern transformer-era activations (GELU, Swish) plotted
     alongside the classics, since most explainers still stop at
     ReLU despite GELU and Swish being what GPT- and BERT-style
     models actually use in 2026.

Author: Khalid Hussain, ReviewPublically.com
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Activation functions and their exact derivatives
# ---------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    # Exact GELU using the Gaussian CDF (erf-based), not the tanh approximation
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_grad(x):
    from scipy.special import erf
    cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def swish_grad(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

# ---------------------------------------------------------------
# 2. Real gradient values at the extremes (the vanishing gradient
#    problem, shown as numbers instead of described in words)
# ---------------------------------------------------------------
test_points = np.array([-6, -4, -2, 0, 2, 4, 6], dtype=float)

print("Gradient values at increasing input magnitude:")
print(f"{'x':>6} | {'sigmoid':>10} | {'tanh':>10} | {'ReLU':>8} | {'GELU':>10}")
for x in test_points:
    print(f"{x:6.1f} | {sigmoid_grad(x):10.6f} | {tanh_grad(x):10.6f} | {relu_grad(x):8.1f} | {gelu_grad(x):10.6f}")

print(f"\nAt x=6, sigmoid's gradient has shrunk to {sigmoid_grad(6.0):.8f}")
print(f"({sigmoid_grad(0.0) / sigmoid_grad(6.0):.0f}x smaller than its peak gradient at x=0)")
print("This is the exact mechanism behind vanishing gradients in deep sigmoid networks:")
print("stack enough layers like this and the gradient reaching early layers underflows toward zero.")

# ---------------------------------------------------------------
# 3. Dying ReLU, demonstrated with a real (if simplified) update
# ---------------------------------------------------------------
print("\n--- Dying ReLU demonstration ---")
np.random.seed(0)
w = 0.5   # a single neuron's weight, deliberately unlucky initialization
b = 2.0
lr = 0.5
X_batch = np.random.uniform(-1, 1, 20)  # a batch of typical inputs
# dL/dz held constant and positive: gradient descent will keep pushing
# this neuron's pre-activation down every step it is still active,
# the way a persistently large error signal would in a real network.
dL_dz = 3.0

print(f"Initial weight={w}, bias={b}")
for step in range(6):
    z = w * X_batch + b
    grad_b = np.mean(relu_grad(z) * dL_dz)   # 0 once the neuron is fully dead
    b = b - lr * grad_b                       # standard gradient descent
    z_after = w * X_batch + b
    alive_fraction = np.mean(z_after > 0)
    print(f"Step {step+1}: bias={b:.3f}, gradient={grad_b:.3f}, fraction of batch still activating: {alive_fraction:.2f}")

print("\nOnce z <= 0 for the entire batch, ReLU's gradient is 0 everywhere in that")
print("batch, so this neuron stops receiving any weight update. It is now dead,")
print("permanently, since a zero gradient means no future step can revive it either.")

# ---------------------------------------------------------------
# 4. Plots for the article
# ---------------------------------------------------------------
x = np.linspace(-6, 6, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].plot(x, sigmoid(x), label="Sigmoid", linewidth=2)
axes[0].plot(x, tanh(x), label="Tanh", linewidth=2)
axes[0].plot(x, relu(x), label="ReLU", linewidth=2)
axes[0].plot(x, leaky_relu(x), label="Leaky ReLU", linewidth=2, linestyle="--")
axes[0].plot(x, gelu(x), label="GELU", linewidth=2)
axes[0].plot(x, swish(x), label="Swish", linewidth=2, linestyle=":")
axes[0].set_title("Activation Functions")
axes[0].legend()
axes[0].axhline(0, color="gray", linewidth=0.5)
axes[0].axvline(0, color="gray", linewidth=0.5)

axes[1].plot(x, sigmoid_grad(x), label="Sigmoid'", linewidth=2)
axes[1].plot(x, tanh_grad(x), label="Tanh'", linewidth=2)
axes[1].plot(x, relu_grad(x), label="ReLU'", linewidth=2)
axes[1].plot(x, leaky_relu_grad(x), label="Leaky ReLU'", linewidth=2, linestyle="--")
axes[1].plot(x, gelu_grad(x), label="GELU'", linewidth=2)
axes[1].plot(x, swish_grad(x), label="Swish'", linewidth=2, linestyle=":")
axes[1].set_title("Their Derivatives (Gradients)")
axes[1].legend()
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].axvline(0, color="gray", linewidth=0.5)

fig.tight_layout()
fig.savefig("/home/claude/activation-functions-from-scratch/activation_functions.png", dpi=150)
plt.close()

# Dying ReLU visualization
fig, ax = plt.subplots(figsize=(9, 4.5))
steps = list(range(1, 7))
alive_fractions = []
w2, b2 = 0.5, 2.0
for step in steps:
    z = w2 * X_batch + b2
    grad_b = np.mean(relu_grad(z) * dL_dz)
    b2 = b2 - lr * grad_b
    z_after = w2 * X_batch + b2
    alive_fractions.append(np.mean(z_after > 0) * 100)

ax.plot(steps, alive_fractions, marker="o", linewidth=2, color="#B2433B")
ax.set_xlabel("Training step")
ax.set_ylabel("% of batch still activating (z > 0)")
ax.set_title("A ReLU Neuron Dying in Real Time")
ax.set_ylim(-5, 105)
fig.tight_layout()
fig.savefig("/home/claude/activation-functions-from-scratch/dying_relu.png", dpi=150)
plt.close()

print("\nSaved: activation_functions.png, dying_relu.png")
