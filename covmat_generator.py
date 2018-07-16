#conv_extract_generator_class.py

#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

#this class is used to generate the training and validation
#sets used for the MLP autoencoder for the TCGA-GBM data set.
# As features, this will use the SE(n) representation of the covariance matrix,
# where the features are the eigenvectors, eigenvalues, and the SE(n) norm


import numpy as np
import keras
import pydicom
from skimage.transform import resize
from keras.models import load_model
import img_to_cov as itc
import SE3util as SE3



class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=10, dim=(273),
        shuffle=True):
        'Initialization'
        #dim is equal to the 16x16 matrix of eigenvalues flattened
        #plus the 16 eigenvalues plus the norm of the matrix
        #256+16+1=673
        self.dim = dim
        self.imdim = (128,128)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def rgbToGray(self, rgbMat):
        grayMat = np.empty(self.imdim, dtype=np.float32)
        grayMat = 0.2989*rgbMat[:,:,0] + 0.5870*rgbMat[:,:,1] + 0.1140*rgbMat[:,:,2]
        return grayMat

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim), 
            dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim), 
            dtype=np.float32)

        path = '../desktop/cancer/TCGA-GBM/'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            ds = pydicom.dcmread(path+ID)
            temp = ds.pixel_array
            resized = resize(temp, self.imdim, mode='constant')
            if resized.shape == (128,128,3):
                resized = self.rgbToGray(resized)
            if resized.shape != (128, 128, 1) and resized.shape != (128, 128):
                print("unrecognized shape", resized.shape)
            resized = resized.reshape(128,128).astype(np.float32)
            im_cov = itc.im_to_cov(resized) #dim=16x16 (4x4 kernel)
            vec, mat = SE3.CovToMatVec(im_cov) #dim
            evecs = mat.flatten()
            norm = SE3.norm(mat,vec)
            concat = np.hstack((evecs, vec, norm))
            #print(concat.shape)

            X[i,:] = concat
            y[i,:] = X[i,:]



        # Store target
        # y[:] = self.labels[list_IDs_temp] #need to fill with labels

        return X, y


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
