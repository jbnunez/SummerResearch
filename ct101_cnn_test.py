#ct101_cnn_test.py
import numpy as np
import scipy as sc
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D
import ct101_cnn_data as data


num_epochs = 10
batch_size = 10


X_train = data.X_train
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test
N, height, width, channels = X_train.shape
input_shape = (height, width, channels)
num_classes = y_train.shape[1]


print('==>Building Model.')

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), 
                 data_format="channels_last",
                 activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


print('==>Compiling Model.')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


print('==>Fitting Model.')
model.fit(x=X_train, y=y_train, batch_size=batch_size, 
    epochs=num_epochs, validation_split=0.2)


print('==>Evaluating Model.')
print(model.evaluate(x=X_test, y=y_test))






