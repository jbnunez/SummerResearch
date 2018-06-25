#mnist_cnn_data.py
"""

This file is used to load and process mnist data, and is imported

"""
import pandas as pd
import numpy as np
from scipy import random, linalg


print('==>Loading data...')

# create the headers for data frame since original data dodes not have headers
name_list = ['pix_{}'.format(i + 1) for i in range(784)]
name_list = ['label'] + name_list

# read the training data
df_train = pd.read_csv('http://pjreddie.com/media/files/mnist_train.csv', \
	sep=',', engine='python', names = name_list)

# read the test data
df_test = pd.read_csv('http://pjreddie.com/media/files/mnist_test.csv', \
	sep=',', engine='python', names = name_list)

print('==>Data loaded succesfully.')


# Process training data, so that X (pixels) and y (labels) are seperated
X_train = np.array(df_train[:][[col for col in df_train.columns \
	if col != 'label']]) / 256.
train_len, hw = X_train.shape
X_train = X_train.reshape((train_len,28,28,1))

y_train2 = np.array(df_train[:][['label']])


# Process test data, so that X (pixels) and y (labels) are seperated
X_test = np.array(df_test[:][[col for col in df_test.columns \
	if col != 'label']]) / 256.
test_len, hw = X_test.shape
X_test = X_test.reshape((test_len,28,28,1))
#print(X_test.shape)
y_test2 = np.array(df_test[:][['label']])



m, height, width, dummy= X_train.shape
test_m, test_height, test_width, dummy = X_test.shape


#turn into one-hot vectors
y_train = np.zeros((m, 10))
for i in range(m):
	y_train[i, y_train2[i]] = 1.

y_test = np.zeros((test_m, 10))
for i in range(test_m):
	y_test[i, y_test2[i]] = 1.

print('==>Data Relabeled succesfully.')


