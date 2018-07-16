#keras_generator_class.py
#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#this class is used to generate the training and validation sets used for the 
#auto-encoder for the TCGA-GBM data set

import numpy as np
import keras
import pydicom
from skimage.transform import resize


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=10, dim=(128,128),
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

    def rgbToGray(self, rgbMat):
        grayMat = np.empty((*self.dim, self.n_channels), 
            dtype=np.float32)
        r,g,b = rgbMat[:,:,0], rgbMat[:,:,1], rgbMat[:,:,2]
        r,g,b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
        grayMat[:,:,0] = 0.2989*r + 0.5870*g + 0.1140*b
        return grayMat

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), 
            dtype=np.float32)#np.uint8)
        y = np.empty((self.batch_size, 128*128), 
            dtype=np.float32)#np.uint8)

        path = '../desktop/cancer/TCGA-GBM/'
        zer = np.zeros(128*128)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            ds = pydicom.dcmread(path+ID)
            temp = ds.pixel_array
            # if not np.any(temp):
            #     print("oh fuck all zeros")
            # print(np.max(temp.flatten()))
            resized = resize(temp, self.dim, mode='constant')#, preserve_range=True)
            # if not np.any(resized):
            #     print("oh god oh not all zeros jesus christ")
            if resized.shape == (128,128,3):
                resized = self.rgbToGray(resized)
            if resized.shape != (128, 128, 1) and resized.shape != (128, 128):
                print("unrecognized shape", resized.shape)
            # if not np.any(resized):
            #     print("fuck me ")
            resized = resized.reshape(128,128).astype(np.float32)#.astype(np.uint8)
            # if not np.any(resized):
            #     print("oh son of a bitch ")
            X[i,:,:,:] = resized.reshape(128,128,1)
            # if not np.any(X[i,:,:,:]):
            #     print("eat a motherfucking dick ")
            y[i,:] = resized.reshape(128*128)
            # print(np.max(y))
            # if not np.any(y[i,:]):
            #     print("oops all zeros")
            #y[i] = channeled.reshape(256,256).flatten()


        # Store target
        #y[:] = X[:].flatten()

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
