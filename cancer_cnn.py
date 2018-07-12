#cancer_cnn.py

import numpy as np
import scipy as sc
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D
import cancer_cnn_data as data


num_epochs = 20
batch_size = 5


X = data.images
N, height, width, channels = X.shape
y = data.targets
input_shape = (height, width, channels)
num_classes = y.shape[1]


print('==>Building Model.')

model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


print('==>Compiling Model.')
model.compile(loss='mean_squared_error', optimizer='adam')


print('==>Fitting Model.')
model.fit(x=X, y=y, batch_size=batch_size, 
    epochs=num_epochs, validation_split=0.1)


print('==>Evaluating Model.')
print(model.evaluate(x=X, y=y))