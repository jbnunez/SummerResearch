#ct101_data_split_xtrain.py
import numpy as np
X = np.load("images.npy")
inds = np.load("ct101_testinds.npy")
# N, d1, d3, d3= X.shape
# N2 = inds.shape[0]
# train = np.zeros((N-N2, d1, d3, d3))
# j = 0
# for i in range(len(N)):
# 	if i 

np.save("X_train_ct101", np.delete(X, inds, axis=0))
