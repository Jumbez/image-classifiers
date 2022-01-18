# Jukka-Pekka Keinänen


import tensorflow as tf
from tensorflow.keras import models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import pickle 
import numpy as np

# One hot encode for classes, each row represent a single class.
ENCODED_CLASSES = np.identity(10, dtype=np.int64)

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Loads all batches
# For testing change the paths
datadict_tr1 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict_tr2 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict_tr3 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict_tr4 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict_tr5 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_5')
datadict_test = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/test_batch')

# Makes one big array of the training data
X_tr = np.concatenate((datadict_tr1["data"], datadict_tr2["data"], datadict_tr3["data"], datadict_tr4["data"], datadict_tr5["data"]))
Y_tr = np.concatenate((datadict_tr1["labels"], datadict_tr2["labels"], datadict_tr3["labels"], datadict_tr4["labels"], datadict_tr5["labels"]))
# Scales the data to be between 0 and 1
X_tr = X_tr / 255

X_test = datadict_test["data"]
# Scales the data to be between 0 and 1
X_test = X_test / 255
Y_test = datadict_test["labels"]

def class_acc(pred, gt):
    right_label = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            right_label += 1
    return right_label/len(pred)


# Encodes the classes from training data
Y_tr_enc = np.array((ENCODED_CLASSES[Y_tr[0]]))

for i in range(1, Y_tr.shape[0]):
    Y_tr_enc = np.vstack((Y_tr_enc, ENCODED_CLASSES[Y_tr[i]]))
    

print(X_tr.shape)
print(Y_tr_enc.shape)

# neural network stuff
model = Sequential()

model.add(Dense(100, input_dim=3072, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
opt = keras.optimizers.SGD(lr=0.5)
model.compile(optimizer=opt, loss="mse", metrics=["mse"])

model.fit(X_tr, Y_tr_enc, epochs=150, verbose=1)

X_test_pred = model.predict(X_test)
label_list = []

for prediction in X_test_pred:
    label_list.append(np.argmax(prediction))

print(f"Accuracy for this neural network: {class_acc(label_list, Y_test)}")
