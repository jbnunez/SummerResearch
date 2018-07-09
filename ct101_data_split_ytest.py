#split_ct101_data_split_ytest.py
import numpy as np
y = np.load("targets.npy")
inds = np.load("ct101_testinds.npy")
np.save("y_test_ct101", y[inds])
