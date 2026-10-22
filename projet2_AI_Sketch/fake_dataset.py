"""fake_dataset.py — TEMPORAIRE, à supprimer après tests."""
import numpy as np
from pathlib import Path

# 8 classes, 500 exemples chacune, images aléatoires 28x28
NUM_CLASSES = 8
N_PER_CLASS = 500
N_TOTAL = NUM_CLASSES * N_PER_CLASS

X = np.random.rand(N_TOTAL, 28, 28, 1).astype("float32")
y = np.eye(NUM_CLASSES, dtype="float32")[np.repeat(np.arange(NUM_CLASSES), N_PER_CLASS)]

n = len(X)
idx = np.random.permutation(n)
X, y = X[idx], y[idx]

n_train = int(0.7 * n)
n_val = int(0.15 * n)

Path("data/processed").mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    "data/processed/dataset_sketch.npz",
    X_train=X[:n_train], y_train=y[:n_train],
    X_val=X[n_train:n_train+n_val], y_val=y[n_train:n_train+n_val],
    X_test=X[n_train+n_val:], y_test=y[n_train+n_val:],
)
print("Dataset factice créé.")