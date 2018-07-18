#cancer_cnn_classifier.py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D
from cancer_cnn_generator import DataGenerator
import pickle
from  sklearn.model_selection import train_test_split
import h5py




num_epochs = 6
steps_per_epoch = 5000

input_shape = (128,128,1)

IDs = pickle.load(open("cancer_IDs", "rb"))
train_IDs, test_IDs = train_test_split(IDs, test_size=0.2)
train_IDs, val_IDs = train_test_split(train_IDs, test_size=0.15)


params = {'dim': input_shape, 'batch_size': 30 , 'shuffle': True}

label_key = 'patient_tss'
# label_key = 'patient_ts'
# label_key = 'patient_vs'
# label_key = 'patient_dd'
# label_key = 'nte_nnet'
# label_key = 'followup_ts'
# label_key = 'followup_vs'
# label_key = 'followup_dd'


train_generator = DataGenerator(train_IDs, label_key, **params)
val_generator = DataGenerator(val_IDs, label_key, **params)
test_generator = DataGenerator(test_IDs, label_key, **params)

num_classes = train_generator.get_num_classes()

print('==>Building Model.')

model = Sequential()
model.add(Conv2D(32, kernel_size=(8, 8), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


print('==>Compiling Model.')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


print('==>Fitting Model.')
model.fit_generator(generator=train_generator, 
    steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
    validation_data=val_generator, validation_steps=500,
    use_multiprocessing=True, workers=4)

print('==>Evaluating Model.')
print(model.evaluate_generator(generator=test_generator, 
    workers=4, use_multiprocessing=True))


model.save("cancer_cnn_classifier.h5")

