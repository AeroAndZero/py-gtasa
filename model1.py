import tensorflow
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

def multi1():
    net = input_data(shape=[None, 6], name='input')

    #net = fully_connected(net, 5, activation='softsign')
    #net = dropout(net, 0.8)

    net = fully_connected(net, 32, activation='leakyrelu')
    #net = dropout(net, 0.8)

    net = fully_connected(net, 128, activation='leakyrelu')
    #net = dropout(net, 0.8)

    net = fully_connected(net, 128, activation='leakyrelu')
    #net = dropout(net, 0.8)

    #net = fully_connected(net, 32, activation='linear')
    #net = dropout(net, 0.8)

    net = fully_connected(net, 5, activation='softmax')

    net = regression(net, optimizer='momentum', learning_rate=1e-3, name="targets", loss="categorical_crossentropy")

    model = tflearn.DNN(net)

    return model