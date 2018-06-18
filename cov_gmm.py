#cov_gmm.py
"""
Adapted from Starter file for k-means(hw8pr1) of Big Data Summer 2017

The file is seperated into two parts:
    1) the helper functions
    2) the main driver.

Note:
1. In the functions below,  
    1) Let m be the number of samples
    2) Let n be the number of features
    3) Let k be the number of clusters

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from sklearn.metrics.pairwise import rbf_kernel
from scipy import random, linalg, stats
#import stock_data as data

#########################################
#            Helper Functions           #
#########################################

# #log euclidean metric
# metric = "leu"
# #affine invariant metric
# #metric = "aff"

# if metric == "leu":
#     import cov_util as met
# elif metric == "aff":
#     import cov_aff_inv_util as met
# else:
#     raise ValueError("unrerecognized metric")

class spd_k_means():

    def __init__(self, k=5, metric = 'aff'):
        self.K = k
        if metric == "leu":
            import cov_util as met
        elif metric == "aff":
            import cov_aff_inv_util as met
        else:
            raise ValueError("unrerecognized metric")
        #list of cluster centers
        self.clusters = None
        #list of sample cluster assignments
        self.labels = None
        #cost list as means are found
        self.cost_list = None
        #class predictions for clusters
        self.predictions = None

    def fit(self, X, y, eps=1e-4, max_iter=1000, print_freq=5):
        """ This function takes in the following arguments:
            1) X, the data matrix with dimension m x n x n
            2) y, the data labels
            3) eps, the threshold of the norm of the change in clusters
            4) max_iter, the maximum number of iterations
            5) print_freq, the frequency of printing the report

            This function returns the following:
            1) clusters, a list of clusters with dimension k x 1
            2) label, the label of cluster for each data with dimension m x 1
            3) cost_list, a list of costs at each iteration
        """
        m, n, n2 = X.shape
        cost_list = []
        t_start = time.time()
        # randomly generate k clusters
        
        start_inds = np.random.randint(m, size=self.K)
        self.clusters = X[start_inds]
       

        self.label = np.zeros((m, 1)).astype(int)
        iter_num = 0

        while iter_num < max_iter:

            prev_clusters = copy.deepcopy(self.clusters)
            #assign labels
            for i in range(m):
                data = X[i,:,:]
                #diff = data - clusters
                diff = np.array([met.dist(data, cluster) for cluster in self.clusters])

                curr_label = np.argmin(diff)
                self.label[i] = curr_label

            #update means
            for cluster_num in range(self.K):
                ind = np.where(self.label == cluster_num)[0]
                if len(ind)>0:
                    self.clusters[cluster_num,:] = met.cov_mean(X[ind])
                else:
                    print("empty cluster found, reinitializing cluster")
                    self.clusters[cluster_num] = X[np.random.randint(m)]

            # TODO: Calculate cost and append to cost_list
            cost = self.k_means_cost(X,)
            self.cost_list.append(cost)

            if (iter_num + 1) % print_freq == 0:
                print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
            dec = np.array([met.dist(prev_clusters[i], self.clusters[i]) for i in range(self.K)])
            if np.linalg.norm(dec) <= eps:
                print('-- Algorithm converges at iteration {} \
                    with cost {:4.4E}'.format(iter_num + 1, cost))
                break
            if len(self.cost_list)>1:
                if self.cost_list[-1]/self.cost_list[-2]>1.001:
                    print('-- ERROR: cost increased at iteration {} \
                          with cost {:4.4E}'.format(iter_num + 1, cost))
                    break
            iter_num += 1

        print('-- Generating predictions for clusters')
        self.assign_labels(y)

        t_end = time.time()
        print('-- Time elapsed: {t:2.2f} \
            seconds'.format(t=t_end - t_start))
        #return clusters, label, cost_list


    def k_means_cost(self, X):
        """ This function takes in the following arguments:
                1) X, the data matrix with dimension m x n
                2) clusters, the matrix with dimension k x 1
                3) label, the label of the cluster for each data point with
                    dimension m x 1

            This function calculates and returns the cost for the given data
            and clusters.

            NOTE:
                1) The total cost is defined by the sum of the l2-norm difference
                between each data point and the cluster center assigned to this data point
        """
        m, n, n2 = X.shape
        k = self.clusters.shape[0]

        # TODO: Calculate the total cost
        X_cluster = self.clusters[self.label.flatten()]
        diff = np.array([met.dist(X[i], X_cluster[i]) for i in range(m)])
        #cost = np.linalg.norm(diff) ** 2
        cost = np.sum(diff)
        
        return cost


    def assign_labels(self, targets):
        self.predictions = np.zeros(self.K)
        for cluster_num in range(self.K):
            ind = np.where(self.labels == cluster_num)[0]
            pred = stats.mode(targets[ind])
            self.predictions[cluster_num] = pred


    def predict(self, X_test):
        m = X_test.shape[0]
        predictions = np.zeros(m)
        for i in range(m):
            data = X_test[m]
            diff = np.array([met.dist(data, cluster) for cluster in self.clusters])
            cluster_num = np.argmin(diff)
            preds[i] = self.predictions[cluster_num]
        return preds

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





