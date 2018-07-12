#feat_extraction_cnn.py
#based on http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/

import numpy as np
import scipy as sc
import keras
from keras import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D
import pickle
from keras_generator_class import DataGenerator
from  sklearn.model_selection import train_test_split


#use a mutlilayer cnn pooling strategy
encoding_dim = 100
input_shape = (256,256,1)
target_dim = input_shape[0]*input_shape[1]
num_epochs = 5
steps_per_epoch = 10000


#get IDs for training and validation samples
IDs = pickle.load(open("cancer_IDs", "rb"))
train_IDs, val_IDs = train_test_split(IDs, test_size=0.2)
partition = {'train': train_IDs, 'validation': val_IDs}

print("==> Loaded "+str(len(train_IDs))+" training samples")
print("==> Loaded "+str(len(val_IDs))+" validation samples")

labels = None #input and output are identical for autoencoder


# Parameters
params = {'dim': (256, 256), 'batch_size': 30,
    'n_channels': 1, 'shuffle': True}


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


input_image = Input(shape=(input_shape))

#architecture of the encoder
encoded = Conv2D(32, kernel_size=(8, 8), strides=(1, 1), 
    activation='relu')(input_image)
encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoded)
encoded = Conv2D(16, kernel_size=(8, 8), strides=(1, 1), 
    activation='relu')(encoded)
encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
encoded = Conv2D(8, kernel_size=(8, 8), strides=(1, 1), 
    activation='relu')(encoded)
encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
encoded = Conv2D(4, kernel_size=(8, 8), strides=(1, 1)
    , activation='relu')(encoded)
encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
encoded = Flatten()(encoded)
encoded = Dense(400, activation='relu')(encoded)
encoded = Dense(200, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

#maps ecoded data to original
decoded = Dense(target_dim, activation='sigmoid')(encoded)

#maps input to its reconstruction
autoencoder = Model(input_image, decoded)

#maps input to encoded representation
encoder = Model(input_image, encoded)

#placeholder for encoded input
encoded_input = Input(shape=(encoding_dim,))

#retrieve last layer of autoencoder model
decoder_layer = autoencoder.layers[-1]

#decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


print('==>Compiling autoencoder.')
autoencoder.compile(loss='mean_squared_error', optimizer='adam')


print('==>Fitting autoencoder.')
autoencoder.fit_generator(generator=training_generator, 
    steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
    validation_data=validation_generator, validation_steps=100)
    #,use_multiprocessing=True, workers=2)

encoder.save("cancer_encoder")
decoder.save("cancer_decoder")
