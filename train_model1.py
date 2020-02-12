from model1 import multi1
import numpy as np

model = multi1()
MODEL_NAME = "Multi Layer Perceptron 1"

data = np.load("training_data-balanced.npy", allow_pickle=True)

train = data[:-100]
test = data[-100:]

x = np.array([i[0] for i in train])
y = np.array([j[1] for j in train])

test_x = np.array([i[0] for i in test])
test_y = np.array([j[1] for j in test])

model.fit({'input' : x} , {'targets': y} , validation_set=({'input' : test_x} , {'targets': test_y}), n_epoch=15, batch_size=10, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

