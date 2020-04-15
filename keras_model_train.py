from keras_model import keras_multi
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam


model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu', name = 'input') )
#model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))



model.add(Dense(5, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
                        optimizer = 'adam',
                        metrics = ['accuracy'])

#model = keras_multi()

MODEL_NAME = 'KERAS_MLP'
data = np.load('training_data_v3.npy', allow_pickle=True)

train = data[:7000]
test = data[7000:]
X = np.array([i[0] for i in train])
Y = np.array([j[1] for j in train])

test_x = np.array([i[0] for i in test])
test_y = np.array([j[1] for j in test])

model.fit(X,Y,
          epochs = 40,
          batch_size = 5,
          validation_data = (test_x,test_y))

model.save('keras_model_test.model')