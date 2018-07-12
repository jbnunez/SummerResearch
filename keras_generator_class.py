#keras_generator_class.py
#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


import numpy as np
import keras
from scipy import misc



class DataGenerator(keras.utils.Sequence)

    def __init__(self, list_IDs, labels, batch_size=10, dim=(256,256),
        n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), 
            dtype=uint8)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), 
            dtype=uint8)

        path = '../desktop/cancer/TCGA-GBM/'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            temp = misc.imread(path+ID)
            resized = misc.imresize(temp, self.dim)
            X[i] = resized.astype(uint8)


        # Store class
        y[i] = X[i].flatten()

        return X, y

