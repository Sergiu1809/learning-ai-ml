from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

print("Downloading MNIST (cached after first run)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

X = X / 255.0

y = y.astype(int)

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot