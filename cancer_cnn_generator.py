#cancer_cnn_generator.py
#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#this class is used to generate the training and validation sets used for the 
#auto-encoder for the TCGA-GBM data set

import numpy as np
import keras
import pydicom
from skimage.transform import resize
from keras.models import load_model
#import img_to_cov as itc
#import SE3util as SE3
import cancer_labels

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, label_key, batch_size=30, dim=(128,128), 
        shuffle=True):
        'Initialization'
        #dimension is equal to the 400 features from the convolutional encoder
        #plus 200 features from the covariance matrix encoder
        self.dim = dim
        self.batch_size = batch_size
        self.labels = cancer_labels.label_options[label_key]
        self.imdim = (128,128)
        self.target_dim = self.labels[self.labels.index[0]].shape[0]
        self.list_IDs = self.filter_IDs(list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()

    def get_num_classes(self):
        return self.target_dim

    def filter_IDs(self, list_IDs):
        keep = self.labels.index
        return list(filter(lambda x: x[1:13] in keep, list_IDs))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def rgbToGray(self, rgbMat):
        grayMat = np.empty(self.dim, dtype=np.float32)
        r,g,b = rgbMat[:,:,0], rgbMat[:,:,1], rgbMat[:,:,2]
        r,g,b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
        grayMat[:,:,0] = 0.2989*r + 0.5870*g + 0.1140*b
        return grayMat


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim)
        # Initialization
        conv_input = np.empty((self.batch_size, 128,128,1), dtype=np.float32)
        covmat_input = np.empty((self.batch_size, 273), dtype=np.float32)

        X = np.empty((self.batch_size, *self.dim), 
            dtype=np.float32)
        y = np.empty((self.batch_size, self.target_dim), 
            dtype=np.float32)

        path = '../desktop/cancer/TCGA-GBM/'

        shortened_IDs = [i[1:13] for i in list_IDs_temp]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            ds = pydicom.dcmread(path+ID)
            temp = ds.pixel_array
            resized = resize(temp, self.imdim, mode='constant')#, preserve_range=True)
            if resized.shape == (128,128,3):
                resized = self.rgbToGray(resized)
            if resized.shape != (128, 128, 1) and resized.shape != (128, 128):
                print("unrecognized shape", resized.shape)
            X[i,:,:,:] = resized.reshape(1,128,128,1)

            y[i] = self.labels[shortened_IDs[i]]


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
