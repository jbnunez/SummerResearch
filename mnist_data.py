#mnist_data.py
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
X_train = X_train.reshape((train_len,28,28))

y_train = np.array(df_train[:][['label']])


# Process test data, so that X (pixels) and y (labels) are seperated
X_test = np.array(df_test[:][[col for col in df_test.columns \
	if col != 'label']]) / 256.
test_len, hw = X_test.shape
X_test = X_test.reshape((test_len,28,28))
#print(X_test.shape)
y_test = np.array(df_test[:][['label']])




m, height, width = X_train.shape
test_m, test_height, test_width = X_test.shape

AvNum = 100 # number of images to average from
nbh = 3 # size of each neighborhood
Stp = 2 #step size for neighborhood
Len = len(np.arange(nbh, height, Stp))
print(Len)

X_covs = np.zeros((10, 13, 13, 2, 2))

PixelDist = np.zeros((height, width))
for r in range(height):
    for s in range(width):
        v = [r,s]
        PixelDist[r,s] = np.linalg.norm(v)

for k in range(10):
   
    # Pick Digit for training:
    List = np.where(y_train == k)[0]
    L    = len(List)
    #print(List)
    #print(L)
    Digs = np.random.randint(L, size=AvNum) # AvNum-many random digits taken from the list.
    #print(Digs)
    Inds = sorted(List[Digs])

    # Reorient Images
    Pics = np.zeros((AvNum,height,width))
    PicsVec = np.zeros((AvNum,height*width))
    for h in range(AvNum):
        Pics[h,:,:]  = X_train[Inds[h],:,:]#.reshape((height, width))]
        PV           = Pics[h,:,:].T
        PicsVec[h,:] = PV.flatten()
    
    AvCovs = np.zeros((Len,Len,2,2))
    for i in range(Len):
        for j in range(Len):
            print('Averaging neighborhood '+str(i)+ 'x'+str(j)+' of '+str(Len)+'x'+str(Len))
            NeighCovs = np.zeros((AvNum,2,2))
            Dists     = np.zeros(AvNum)
            for h in range(AvNum):                    
                #print("i", (Stp*i)+nbh)
                #print("j",(Stp*j)+nbh)

                Block     = X_train[h, Stp*i:(Stp*i)+nbh, Stp*j:(Stp*j)+nbh]
                DistBlock = PixelDist[Stp*i:(Stp*i)+nbh, Stp*j:(Stp*j)+nbh]
        
                PixelObs = np.zeros((2, nbh**2))
                PixelObs[0,:] = Block.flatten()
                PixelObs[1,:] = DistBlock.flatten()
                #calculate covariance between location and pixel value
                mat = np.cov(PixelObs)
                if np.isnan(mat[:]).any():
                    #raise ValueError("oh shit it's a nan")
                    print("nan", i, j)
                    mat = np.eye(2)
                    mat[1,1]=1.01
                if np.array_equal(mat, np.zeros((2,2))):
                    mat = 0.01*np.eye(2)
                    mat[1,1]=0.02
                NeighCovs[h,:,:] = mat
            
        
            AvCovs[i,j,:,:] = met.cov_mean(NeighCovs)
            #Find distances
            # for h in range(AvNum):
            #     Dists[h] = met.dist(NeighCorrs[m,:,:], AvCorrs[i,j,:,:])#grad_descent_closest(NeighCorrs[:,:,m], AvCorrs[:,:,i,j])[2]
            
            # MeanStd[0,0,i,j] = np.mean(Dists)
            # MeanStd[1,0,i,j] = np.std(Dists)
    X_covs[k] = AvCovs
# #distance matrix by neighborhood
# dmat = np.zeros((10,10,13,13))
# for i in range(10):
#     for j in range(10):
#         for k in range(13):
#             for l in range(13):
#                 dmat[i,j,k,l] = met.dist(X_covs[i,k,l,:,:], X_covs[j,k,l,:,:])
# print(dmat)
    
    