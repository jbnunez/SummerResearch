#covariance k-means
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
from scipy import random, linalg
import mnist_data as data

#########################################
#            Helper Functions           #
#########################################

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

def k_means(X, k, eps=1e-10, max_iter=1000, print_freq=10):
    """ This function takes in the following arguments:
            1) X, the data matrix with dimension m x n x n
            2) k, the number of clusters
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
    
    clusters = []
    for i in range(k):
        A = np.random.rand(n,n)
        clusters.append(A @ A.T)
    clusters = np.array(clusters)

    #clusters = np.random.multivariate_normal((.5 + np.random.rand(n)) * X.mean(axis=0), 10 * X.std(axis=0) * np.eye(n), \
    #   size=k)
    label = np.zeros((m, 1)).astype(int)
    iter_num = 0

    while iter_num < max_iter:

        prev_clusters = copy.deepcopy(clusters)
        #assign labels
        for i in range(m):
            data = X[i,:,:]
            #diff = data - clusters
            diff = np.array([met.dist(data, cluster) for cluster in clusters])

            #curr_label = np.argsort(np.linalg.norm(diff, axis=1)).item(0)
            curr_label = np.argmin(np.linalg.norm(diff, axis=1))
            label[i] = curr_label

        #update means
        for cluster_num in range(k):
            ind = np.where(label == cluster_num)[0]
            if len(ind)>0:
                clusters[cluster_num,:] = met.cov_mean(X[ind])

        # TODO: Calculate cost and append to cost_list
        cost = k_means_cost(X, clusters, label)
        cost_list.append(cost)

        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
        dec = np.array([met.dist(prev_clusters[i], clusters[i]) for i in range(k)])
        if np.linalg.norm(dec) <= eps:
            print('-- Algorithm converges at iteration {} \
                with cost {:4.4E}'.format(iter_num + 1, cost))
            break
        iter_num += 1

    t_end = time.time()
    print('-- Time elapsed: {t:2.2f} \
        seconds'.format(t=t_end - t_start))
    return clusters, label, cost_list

def k_means_cost(X, clusters, label):
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
    k = clusters.shape[0]

    # TODO: Calculate the total cost
    X_cluster = clusters[label.flatten()]
    diff = np.array([met.dist(X[i], X_cluster[i]) for i in range(m)])
    cost = np.linalg.norm(diff) ** 2

    return cost


def grad_descent_closest(S, M):
    n = S.shape.item(0)

    D = np.eye(n)

    tan = np.zeros((n, n)); 
    tan[0,0] = 1 

    step = 0.01 # Step size for gradient descent
    
    def Geo(Pos,Tan): 
        sqr = sc.linalg.sqrtm(Pos)
        inner1 = np.divide(Tan, sqr)
        inner2 = np.divide(inner1, sqr)
        return sqr*sc.linalg.expm(-step*inner2)*sqr

    def norm2(Pos,Tan):
        div = np.divide(Tan, Pos)
        return math.sqrt(np.trace(div @ div))

    def DiagGrad(Pos):
        # Supporting Gradient function
        LogSeq = np.zeros((K,n,n))
        for k in range(K):
            LogSeq[k,:,:] = sc.linalg.logm(np.divide(Pos, DataSeq[k,:,:]))
        
        return Pos*np.sum(LogSeq, axis=0)

    # Intiate While Loop, Determine Minimum

    count = 0 # % Keep track of iterations

    while norm2(D,tan) >= 0.1:
        D = Geo(D,tan);
        tan = DiagGrad(S, M, D);
        count += 1;

    sqrS = sc.linalg.sqrtm(S)
    Opt = pos*M*pos
    inn1 = np.divide(Opt, sqrS)
    inn2 = np.divide(inn1, sqrS)

    Dist = np.norm(sc.linalg.logm(inn2))

    Iters = count

    return D, Opt, Dist, Iters





###########################################
#           Main Driver Function          #
###########################################


if __name__ == '__main__':
    # =============STEP 0: LOADING DATA=================
    print('==> Step 0: Loading data...')
    # Read data
    # path = '5000_points.csv'
    # columns = ['x', 'space', 'y']
    # features = ['x', 'y']
    # df = pd.read_csv(path, sep='  ', names = columns, engine='python')
    # X = np.array(df[:][features]).astype(int)


    df_train = data.df_train
    df_test = data.df_test
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test


    X = data.X_covs


    # =============Implementing K-MEANS=================
    # Fill in the code in k_means() and k_means_cost()
    # NOTE: You may test your implementations by running k_means(X, k)
    #       for any reasonable value of k

    # =============FIND OPTIMAL NUMBER OF CLUSTERS=================
    # Calculate the cost for k between 1 and 20 and find the k with
    #       optimal cost
    print('==> Step 1: Finding optimal number of clusters...')
    cost_k_list = []
    for k in range(1, 21):
        # Get the clusters, labels, and cost list for different k values
        clusters, label, cost_list = k_means(X, k)
        cost = cost_list[-1]
        cost_k_list.append(cost)

        print('-- Number of clusters: {} - cost: {:.4E}'.format(k, cost))

    opt_k = np.argmin(cost_k_list) + 1
    print('-- Optimal number of clusters is {}'.format(opt_k))

    # TODO: Generate plot of cost vs k
    cost_vs_k_plot, = plt.plot(range(1, 21), cost_k_list, 'g^')
    opt_cost_plot, = plt.plot(opt_k, min(cost_k_list), 'rD')    

    plt.title('Cost vs Number of Clusters')
    plt.savefig('kmeans_1.png', format='png')
    plt.close()

    # =============VISUALIZATION=================
    # Generate visualization on running k-means on the optimal k value
    # Be sure to mark the cluster centers from the data point

    # clusters, label, cost_list = k_means(X, opt_k)
    # X_cluster = clusters[label.flatten()]
    # data_plot, = plt.plot(X[:, 0], X[:, 1], 'bo')
    # center_plot, = plt.plot(X_cluster[:, 0], X_cluster[:, 1], 'rD')


    # # set up legend and save the plot to the current folder
    # plt.legend((data_plot, center_plot), \
    #   ('data', 'clusters'), loc = 'best')
    # plt.title('Visualization with {} clusters'.format(opt_k))
    # plt.savefig('kmeans_2.png', format='png')
    # plt.close()
