# Jukka-Pekka Keinänen

import pickle
import numpy as np
import random
from scipy.spatial.distance import cityblock

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Manhattan distance
def cifar10_classifier_1nn(x, trdata, trlabels):
    distance_list = []
    for i in range(trdata.shape[0]):
        distance_list.append(cityblock(x, trdata[i]))

    return trlabels[np.argmin(distance_list)]

def cifar10_classifier_knn(x, trdata, trlabels, k=1):
    distance_list = []
    for i in range(trdata.shape[0]):
        dist = cityblock(x, trdata[i])

        distance_list.append(cityblock(x, trdata[i]))

    return trlabels[np.argmin(distance_list)]
    

def class_acc(pred, gt):
    right_label = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            right_label += 1
    return right_label/len(pred)

# Returns random class, so the accuracy for this classifier should be aroun 10%
def cifar10_classifier_random(x):
    return random.randint(0,9)

# Loads all batches
# For testing, change the paths.
datadict_tr1 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict_tr2 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict_tr3 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict_tr4 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict_tr5 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_5')
datadict_test = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/test_batch')

# Makes one big array of the training data
X_tr = np.concatenate((datadict_tr1["data"], datadict_tr2["data"], datadict_tr3["data"], datadict_tr4["data"], datadict_tr5["data"]))
Y_tr = np.concatenate((datadict_tr1["labels"], datadict_tr2["labels"], datadict_tr3["labels"], datadict_tr4["labels"], datadict_tr5["labels"]))

# Change datatype
X_tr = X_tr.astype(np.int64)

X_test = datadict_test["data"]
Y_test = datadict_test["labels"]
# Change the datatype for calculations
X_test = X_test.astype(np.int64)

# Run random classifier for every test sample (2.3)
random_predicted_labels  = []
for pic in X_test:
    random_predicted_labels.append(cifar10_classifier_random(pic))
print(f"Class accuracy for random classifier: {class_acc(random_predicted_labels, Y_test)}")

# Run 1NN-classifer for every test sample (2.4)
def main():
    predicted_labels = []
    for pic in X_test:
        predicted_labels.append(cifar10_classifier_1nn(pic, X_tr, Y_tr))
    print(f"Class accuracy for 1NN-classifier: {class_acc(predicted_labels, Y_test)} for {counter} images")
    return class_acc(predicted_labels, Y_test)

main()
