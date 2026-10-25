import numpy as np

rng = np.random.default_rng(42)

# A fake batch of 32 MNIST images
X = rng.normal(size=(32, 784))

# a = np.array([[1, 2, 3],
#               [4, 5, 6]])

# print(a.shape)

# rng.normal(size=(2, 3))

# print(np.zeros((2, 3)))
# print(np.zeros(5))

# print(X.shape)


# TODO 1: create W1 and b1 for a layer with 128 neurons.
#         Use rng.normal(size=...) for W1 and np.zeros(...) for b1.

W1 = rng.normal(size=(784, 128))
b1 = np.zeros(128)

# TODO 2: compute Z1 = X @ W1 + b1

Z1 = X @ W1 + b1

# TODO 3: print the shape of X, W1, b1 and Z1.
#         Before you run it, write your prediction on paper.

print(X.shape)
print(W1.shape)
print(b1.shape)
print(Z1.shape)
assert Z1.shape == (32, 128)

# TODO 4: add a second layer, 128 -> 10. Compute Z2.
#         Predict its shape first, then print it.

W2 = rng.normal(size=(128, 10))
b2 = np.zeros(10)
Z2 = Z1 @ W2 + b2

print(Z1.shape)
print(W2.shape)
print(b2.shape)
print(Z2.shape)

# TODO 5: deliberately break it. Build W_bad with shape (128, 784)
#         and try X @ W_bad inside a try/except ValueError.
#         Print the error message. Read it carefully: NumPy tells you
#         exactly which dimensions clashed.

W_bad = rng.normal(size=(128, 784))
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 784)

try:
    Z_bad = X @ W_bad + b1
except ValueError as e:
    print("Caught:", e)


# TODO 6: verify the dot product claim by hand.
#         Take example 0 and neuron 0:
#           manual = np.sum(X[0] * W1[:, 0]) + b1[0]
#         Compare it to Z1[0, 0] using np.isclose(). Print the result.
#         If this prints True, you have proved to yourself that
#         matrix multiplication is just many dot products at once.

manual = np.sum(X[0] * W1[:, 0]) + b1[0]
print(manual)
print(np.isclose(manual, Z1[0, 0]))
