#test_encoder.py
import numpy as np
import scipy as sc
import keras
from keras import Model
from keras.models import load_model, Sequential
from keras.layers import Input
import pickle
from keras_generator_class import DataGenerator
import h5py

input_shape = (128, 128, 1)

#input layer
input_image = Input(shape=(input_shape))

#load model for the encoder created by the autoencoder
# encoder = load_model('cancer_encoder.h5')
# decoder = load_model('cancer_decoder.h5')
# autoencoder = load_model('cancer_autoencoder.h5')


# encoder = load_model('cancer_encoder_100.h5')
# decoder = load_model('cancer_decoder_100.h5')
# autoencoder = load_model('cancer_autoencoder_100.h5')


encoder = load_model('cancer_encoder_bce.h5')
decoder = load_model('cancer_decoder_bce.h5')
autoencoder = load_model('cancer_autoencoder_bce.h5')


IDs = pickle.load(open("cancer_IDs", "rb"))

params = {'dim': (128, 128), 'batch_size': 10,
    'n_channels': 1, 'shuffle': True}
labels = None

generator = DataGenerator(IDs, labels, **params)
X = np.empty((300,128,128,1))
y = np.empty((300,128*128))
for i in range(10):
    X[i:i+10,:,:],y[i:i+10,:] = generator.__getitem__(i)
np.set_printoptions(threshold=np.inf)

print(X[50])

encoded = encoder.predict(X)
decoded = decoder.predict(encoded)
autoencoded = autoencoder.predict(X)
loss = np.linalg.norm(y-decoded)
norm = np.linalg.norm(y)
if not np.any(autoencoded) or not np.any(decoded):
    "oh son of a bitch"
print("decoded norm", np.linalg.norm(decoded))
print("loss norm", loss)
print("target norm", norm)
print("loss over norm", loss/norm)
print("maximum of target", np.max(y))
print("maximum of autoencoded", np.max(decoded))

#decoder.predict(generator.__getitem__(0))

#test_model.evaluate_generator(generator=training_generator, steps=1000)




