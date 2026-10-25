import numpy as np


# Seed 0 so the random numbers are the same on every run
# Without a fixed seed you would get different values each time
# and could not compare your output with mine.
rng = np.random.default_rng(seed=0)


# One example with 4 features. Shape (1, 4): one row, four columns
# Tiny on purpose. 784 pixels would make hand-checking impossible.
# It stays 2D, not shape (4, ), so that x @ W works like the real
# batched forward pass. A batch of one.
x = rng.normal(size=(1, 4))

# The layer: 4 inputs, 3 neurons. Shape (4, 3)
# Rows are inputs, columns are neurons, same as always
# Column j holds neuron j's 4 weights.
W = rng.normal(size=(4, 3))

# One bias per neuron, shape (3,). Starts at zero.
# Prints: [0. 0. 0.]
b = np.zeros(3)

# The correct answer for this example, shape (1, 3).
# One number per neuron: we want neuron 0 to output 1
# and the other two to output 0.
# This is a made-up target so there is something to be wrong about.
y = np.array([[1.0, 0.0, 0.0]])

# Takes W as an argument rather than reading the global.
# This is deliberate: in a moment you will call this with a
# slightly modified W to see hoe the loss responds.


def loss(W):
    # The forward pass from block 1.
    # (1, 4) @ (4, 3) -> (1, 3), then + b broadcasts across the row.
    z = x @ W + b

    # z - y: how far off each of the 3 outputs is. Shape (1, 3).
    # ** 2: squares each difference. Makes everything positive, so
    #       errors cannot cancel out, and punishes big misses harder.
    # mp.sum: adds the 3 squared errors into ONE number.
    return np.sum((z-y) ** 2)

    # Why the loss is a single number:
    # - Gradient descent minimises one quantity. ∂loss/∂w is only defined if the loss is scalar.
    # - With several loss values there is no way to say whether a weight change helped.
    # - np.sum((z - y) ** 2) collapses all errors into one scalar.
    # - The squaring matters: it stops positive and negative errors from cancelling, and penalises large errors more than small ones.
    # - This is mean squared error (without the mean), the standard loss for regression


print(loss(W))

# h is the nudge: small enough that the slope is roughly constant
# across it, big enough that floating point can represent the
# difference. ie-5 is the usual compromise.
h = 1e-5

# .copy() makes a separate array. Without it, W_plus and W would be
# the same object in memory, so changing one would change the other
# and the comparison below would be meaningless.
W_plus = W.copy()

# Nudge exactly one weight. Every other weight stays as it was.
# This is what "partial derivative" means in code: move one, hold the rest.
W_plus[0, 0] = W_plus[0, 0] + h

# Same nudge in the other direction.
W_minus = W.copy()
W_minus[0, 0] = W_minus[0, 0] - h

# Rise over run. The loss difference between the two nudged versions,
# divided by the total distance between them, which is 2h
# This is the slope of the loss with respect to W[0, 0]
numeric = (loss(W_plus) - loss(W_minus)) / (2 * h)

print(numeric)

# Recompute the forward pass so z is available outside the loss function
# loss() computes z internally but returns only the number, so we need it again here
z = x @ W + b


# Link 1 times Link 2, the chain rule result.
# 2 * (z[0,0] - y[0,0]) is how much the loss responds to z[0,0]
# x[0, 0] is how much z[0,0] responds to W[0,0]
analytic = 2 * (z[0, 0] - y[0, 0]) * x[0, 0]

# Print both so you can compare them by eye
print("analytic:", analytic)
print("numeric: ", numeric)

# np.isclose allows a tiny difference. They will not match exactly:
# numeric is an approximation using h = 1e-5, so it carries a small error.
# Exact equality would be the suspicious result, not the reassuring one.
print(np.isclose(analytic, numeric))
