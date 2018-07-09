#ct101_data_split_ytrain.py
import numpy as np
y = np.load("targets.npy")
inds = np.load("ct101_testinds.npy")
np.save("y_train_ct101", np.delete(y, inds, axis=0))
