#cancer_img_autoencoder.py
#based on https://blog.keras.io/building-autoencoders-in-keras.html

import numpy as np
import scipy as sc
import keras
from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D
import pickle
from keras_generator_class import DataGenerator
from  sklearn.model_selection import train_test_split
import h5py


#use a mutlilayer cnn pooling strategy
encoding_dim = 400
input_shape = (128,128,1)
target_dim = input_shape[0]*input_shape[1]
num_epochs = 10
steps_per_epoch = 1000


#get IDs for training and validation samples
IDs = pickle.load(open("cancer_IDs", "rb"))
train_IDs, val_IDs = train_test_split(IDs, test_size=0.2)
partition = {'train': train_IDs, 'validation': val_IDs}

print("==> Loaded "+str(len(train_IDs))+" training samples")
print("==> Loaded "+str(len(val_IDs))+" validation samples")

labels = None #input and output are identical for autoencoder


# Parameters
params = {'dim': (128, 128), 'batch_size': 30,
    'n_channels': 1, 'shuffle': True}


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


input_image = Input(shape=input_shape)

#architecture of the encoder
layer1 = Conv2D(32, kernel_size=(8, 8), strides=(1, 1), 
    activation='sigmoid')(input_image)
layer2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer1)
layer3 = Conv2D(16, kernel_size=(8, 8), strides=(1, 1), 
    activation='sigmoid')(layer2)
layer4 = MaxPooling2D(pool_size=(2, 2))(layer3)
layer5 = Conv2D(8, kernel_size=(8, 8), strides=(1, 1), 
    activation='sigmoid')(layer4)
layer6 = MaxPooling2D(pool_size=(2, 2))(layer5)
layer7 = Conv2D(4, kernel_size=(8, 8), strides=(1, 1)
    , activation='sigmoid')(layer6)
layer8 = MaxPooling2D(pool_size=(2, 2))(layer7)
layer9 = Flatten()(layer8)
layer10 = Dense(800, activation='sigmoid')(layer9)
encoded = Dense(encoding_dim, activation='sigmoid')(layer10)

#encoder model
encoder = Model(input_image, encoded)
encoder.summary()


#placeholder for encoded input
encoded_input = Input(shape=(encoding_dim,))

#maps ecoded data to original
#intermed = Dense(400, activation='relu')(encoded_input)
decoded = Dense(target_dim, activation='relu')(encoded_input)

#decoder model
decoder = Model(encoded_input, decoded)
#decoder.summary()

#maps input to its reconstruction
autoencoder = Model(input_image, decoder(encoder(input_image)))
#autoencoder.summary()


print('==>Compiling autoencoder.')
#autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')
autoencoder.compile(loss='mean_squared_error', optimizer='adam')


autoencoder.summary()

print('==>Fitting autoencoder.')
autoencoder.fit_generator(generator=training_generator, 
    steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
    validation_data=validation_generator, validation_steps=100,
    use_multiprocessing=True, workers=4)

#print('==>Making decoder and encoder.')

print('==>Saving decoder and encoder.')
#encoder.summary()
encoder.save("cancer_encoder.h5")
decoder.save("cancer_decoder.h5")
# autoencoder.save("cancer_autoencoder.h5")

# encoder.save("cancer_encoder_bce.h5")
# decoder.save("cancer_decoder_bce.h5")
# autoencoder.save("cancer_autoencoder_bce.h5")

