#ct101_cnn_classes.py
import numpy as np
from scipy import misc

class DataGenerator

    def __init__(self, list_IDs, labels, batch_size=32, dim=(494,708,3), n_channels=3,
             n_classes=101, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        path = '../desktop/ct101/101_ObjectCategories/'
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            temp = misc.imread(path+ID)
            idim1, idim2 = temp.shape[0:2]
            diff1 = self.dim[0]-idim1
            diff2 = self.dim[1]-idim2
            start1 = diff1//2
            start2 = diff2//2
            if len(temp.shape) == 3:
                X[i, start1:start1+idim1, start2:start2+idim2] = temp
            else:
                X[i, start1:start1+idim1, start2:start2+idim2, 0] = temp


        # Store class
        y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

