#keras_generator_class.py
#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


import numpy as np
import keras
from scipy import misc
import pydicom


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=10, dim=(256,256),
        n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
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

    def rgbToGray(rgbMat):
        grayMat = np.empty((*self.dim, self.n_channels), 
            dtype=np.uint8)
        grayMat[:,:,0] = 0.2989*rgbMat[:,:,0] + 0.5870*rgbMat[:,:,1] + 0.1140*rgbMat[:,:,2]
        return grayMat

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), 
            dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), 
            dtype=np.uint8)

        path = '../desktop/cancer/TCGA-GBM/'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            ds = pydicom.dcmread(path+ID)
            temp = ds.pixel_array
            resized = misc.imresize(temp, self.dim)
            if resized.shape == (256,256,3):
                resized = rgbToGray(resized)
            resized = resized.reshape(256,256).astype(np.uint8)
            X[i] = resized.reshape(256,256,1)


        # Store target
        y[i] = X[i].flatten()

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
