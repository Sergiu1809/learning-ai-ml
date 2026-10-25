import numpy as np


# Seed 0 so the random numbers are the same on every run
# Without a fixed seed you would get different values each time
# and could not compare your output with mine.
rng = np.random.default_rng(0)


# One example with 4 features. Shape (1, 4): one row, four columns
# Tiny on purpose. 784 pixels would make hand-checking impossible.
# It stays 2D, not shape (4, ), so that x @ W works like the real
# batched forward pass. A batch of one.
x = rng.normal(size=(1, 4))

# The layer: 4 inputs, 3 neurons. Shape (4, 3)
# Rows are inputs, columns are neurons, same as always
# Column j holds neuron j's 4 weights.
w = rng.normal(size=(4, 3))
