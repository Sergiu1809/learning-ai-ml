from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

print("Downloading MNIST (cached after first run)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

print(f"X shape: {X.shape}")   # should be (70000, 784)
print(f"y shape: {y.shape}")   # should be (70000,)
print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
print(f"Pixel value range: min={X.min()}, max={X.max()}")
print(f"First 10 labels: {y[:10]}")

# visualize the first digit to confirm it looks right
first_image = X[0].reshape(28, 28)
plt.imshow(first_image, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.savefig("sample_digit.png")
print("\nSaved sample_digit.png — open it to check the image looks like a real digit.")
