#split_ct101_data.py
import numpy as np
print("==> Loading data")
X = np.load("images.npy")
inds = np.random.choice(X.shape[0], size=X.shape[0]//5)
print("==> Splitting data")

np.save("ct101_testinds", inds)
np.save("X_test_ct101", X[inds])
