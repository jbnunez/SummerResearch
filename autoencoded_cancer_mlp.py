#autoencoded_cancer_mlp.py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
import cancer_encoded_generator_class as DataGenerator
import pickle



num_epochs = 10
steps_per_epoch = 1000



IDs = pickle.load(open("cancer_IDs", "rb"))
train_IDs, test_IDs = train_test_split(IDs, test_size=0.2)
train_IDs, val_IDs = train_test_split(train_IDs, test_size=0.15)


params = {'dim': (600), 'batch_size': 30 , 'shuffle': True, 
    conv_encoder_path='cancer_encoder.h5', 
    covmat_encoder_path='cancer_cov_encoder.h5'}

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
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


print('==>Compiling Model.')
model.compile(loss='mean_squared_error', optimizer='adam')


print('==>Fitting Model.')
model.fit_generator(generator=training_generator, 
    steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
    validation_data=val_generator, validation_steps=100,
    use_multiprocessing=True, workers=4)

print('==>Evaluating Model.')
print(model.evaluate_generator(generator=test_generator, 
    workers=4, use_multiprocessing=True, verbose=1))




