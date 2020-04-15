from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from keras.optimizers import adam

def keras_multi():
    model = Sequential()

    model.add(Dense(64, input_dim=6, activation='relu', name = 'input') )
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = 'softmax'))
    net = model.compile(loss='categorical_crossentropy',
                        optimizer = 'adma',
                        metrics = ['accuracy'])
    return net