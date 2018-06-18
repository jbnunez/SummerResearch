#cov_knn.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from sklearn.metrics.pairwise import rbf_kernel
from scipy import random, linalg, stats

#log euclidean metric
metric = "leu"
#affine invariant metric
#metric = "aff"

if metric == "leu":
    import cov_util as met
elif metric == "aff":
    import cov_aff_inv_util as met
else:
    raise ValueError("unrerecognized metric")

class spd_knn():


    def __init__(X, y, k=5):#, metric = 'aff'):
        self.K = k
        self.X = X
        self.y = y
        self.X_len = X.shape[0]

        # if metric == "leu":
        #     import cov_util as met
        # elif metric == "aff":
        #     import cov_aff_inv_util as met
        # else:
        #     raise ValueError("unrerecognized metric")


    def predict(self, X_test):
        m = X_test.shape[0]
        predictions = np.zeros(m)
        for i in range(m):
            data = X_test[i]
            #list of nearest neighbors in format [distance, label]
            nearest_dists = np.full(self.K, np.inf)
            nearest_labels = np.zeros(self.K)
            for j in range(self.X_len):
                dist = met.dist(data, self.X[j])
                if dist < np.max(nearest_dists):
                    max_ind = np.argmax(nearest_dists)
                    nearest_dists[max_ind] = dist
                    nearest_labels[max_ind] = self.y[j]

            predictions[i] = stats.mode(nearest_labels)

        return predictions


    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        counter = float(0)
        m = y_test.shape[0]
        for i in range(m):
            if y_test[i] == preds[i]:
                counter += 1
        acc = counter/m
        print('Test accuracy: '+str(100*acc)+' percent accurate')
        return acc


