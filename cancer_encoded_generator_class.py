#cancer_encoded_generator_class.py
#made using source:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#this class is used to generate the training and validation sets used for the 
#auto-encoder for the TCGA-GBM data set

import numpy as np
import keras
import pydicom
from skimage.transform import resize
from keras.models import load_model
import img_to_cov as itc
import SE3util as SE3
import cancer_labels

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, label_key, batch_size=30, dim=(600), 
        conv_encoder_path='cancer_encoder.h5', covmat_encoder_path='cancer_cov_encoder.h5',
        shuffle=True):
        'Initialization'
        #dimension is equal to the 400 features from the convolutional encoder
        #plus 200 features from the covariance matrix encoder
        self.dim = dim
        self.batch_size = batch_size
        self.labels = cancer_labels.label_options[label_key]
        self.target_dim = self.labels[self.labels.index[0]].shape
        self.list_IDs = self.filter_IDs(list_IDs)
        self.n_channels = n_channels
        self.conv_encoder = load_model(conv_encoder_path)
        self.covmat_encoder = load_model(covmat_encoder_path)
        self.shuffle = shuffle
        self.on_epoch_end()

    def get_num_classes(self):
        return self.target_dim[0]

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
        X = np.empty((self.batch_size, *self.dim), 
            dtype=np.float32)#np.uint8)
        y = np.empty((self.batch_size, self.target_dim), 
            dtype=np.float32)#np.uint8)

        path = '../desktop/cancer/TCGA-GBM/'

        shortened_IDs = [i[1:13] for i in list_IDs_temp]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        # Store sample

            ds = pydicom.dcmread(path+ID)
            temp = ds.pixel_array
            resized = resize(temp, self.dim, mode='constant')#, preserve_range=True)
            if resized.shape == (128,128,3):
                resized = self.rgbToGray(resized)
            if resized.shape != (128, 128, 1) and resized.shape != (128, 128):
                print("unrecognized shape", resized.shape)
            resized = resized.reshape(128,128).astype(np.float32)#.astype(np.uint8)
            channeled = resized.reshape(128,128,1)
            conv_encoded = self.conv_encoder.predict(channeled)

            im_cov = itc.im_to_cov(resized) #dim=16x16 (4x4 kernel)
            vec, mat = SE3.CovToMatVec(im_cov) #dim
            evecs = mat.flatten()
            norm = SE3.norm(mat,vec)
            concat = np.hstack((evecs, vec, norm))
            covmat_encoded = self.covmat_encoder.predict(concat)
            
            X[i,:,:,:] = np.hstack((conv_encoded, covmat_encoded))
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
