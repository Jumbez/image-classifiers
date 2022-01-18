# Jukka-Pekka Keinänen

import pickle
from typing import Dict
import numpy as np
from numpy.core.shape_base import vstack
from numpy.lib.function_base import cov
from scipy.stats import norm
from scipy.stats import multivariate_normal
from skimage.transform import resize
from matplotlib import pyplot as plt

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Loads all batches
# For testing on your computer, change the names of the paths
datadict_tr1 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict_tr2 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict_tr3 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict_tr4 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict_tr5 = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/data_batch_5')
datadict_test = unpickle('/home/jumbezz/Koulu/dataml100/cifar-10-python/cifar-10-batches-py/test_batch')

# Makes one big array of the training data
X_tr = np.concatenate((datadict_tr1["data"], datadict_tr2["data"], datadict_tr3["data"], datadict_tr4["data"], datadict_tr5["data"]))
Y_tr = np.concatenate((datadict_tr1["labels"], datadict_tr2["labels"], datadict_tr3["labels"], datadict_tr4["labels"], datadict_tr5["labels"]))
# Reshapes the data for resizing
X_tr = np.reshape(X_tr, (X_tr.shape[0],3,32,32)).transpose(0,2,3,1)

X_test = datadict_test["data"]
Y_test = datadict_test["labels"]

# Reshapes the data for resizing
X_test = np.reshape(X_test, (X_test.shape[0],3,32,32)).transpose(0,2,3,1)

def class_acc(pred, gt):
    right_label = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            right_label += 1
    return right_label/len(pred)

def cifar10_color(X):
    scaled_pics = np.array([])
    # Scales and adds the first picture
    scaled_pics = np.array(resize(X[0], (1, 1, 3), preserve_range=True).reshape(1,3))
    for i in range(1, X.shape[0]):
        scaled_pic = resize(X[i], (1, 1, 3), preserve_range=True).reshape(1, 3)
        scaled_pics = np.vstack((scaled_pics, scaled_pic))
    return scaled_pics

def cifar_10_naivebayes_learn(X_p, Y):
    mu_matrix = np.array([])
    std_matrix = np.array([])
    p_vector = []
    
    for c in range(10):
        class_pictures = np.array([])
        for i in range(X_p.shape[0]):
            pic = X_p[i]
            label = Y[i]
            if label == c:
                if class_pictures.shape[0] !=0:
                    class_pictures = np.vstack((class_pictures, pic))
                else:
                    class_pictures = np.array([pic])
        if c == 0:
            mu_matrix = np.mean(class_pictures, axis=0)
            std_matrix = np.std(class_pictures,axis=0)
            p_vector.append(class_pictures.shape[0] / len(Y))
            
        else:
            mu_matrix = np.vstack((mu_matrix, np.mean(class_pictures, axis=0)))
            std_matrix = np.vstack((std_matrix, np.std(class_pictures, axis=0)))             
            p_vector.append((class_pictures.shape[0] / len(Y)))
            
    return mu_matrix, std_matrix, p_vector

def cifar10_classifier_naivebayes(x, mu, sigma, p):
    class_probabilities = []
    for i in range(10):
        mu_r = mu[i][0]
        mu_g = mu[i][1]
        mu_b = mu[i][2]

        sigma_r = sigma[i][0]
        sigma_g = sigma[i][0]
        sigma_b = sigma[i][0]
        p_c = p[i]

        class_probabilities.append(norm.pdf(x[0], mu_r, sigma_r) * norm.pdf(x[1], mu_g, sigma_g) * norm.pdf(x[2], mu_b, sigma_b) * p_c)
    
    return np.argmax(class_probabilities)


def cifar_10_bayes_learn(Xf, Y, dim=1):
    mu_matrix = np.array([])
    cov_matrix = np.array([])
    p_vector = []
    for c in range(10):
        
        class_pictures = np.array([])
        for i in range(Xf.shape[0]):
            pic = Xf[i]
            label = Y[i]
            if label == c:
                if class_pictures.shape[0] !=0:
                    class_pictures = np.vstack((class_pictures, pic))
                else:
                    class_pictures = np.array([pic])
        
        if c == 0:
            mu_matrix = np.mean(class_pictures, axis=0)
            cov = np.cov(class_pictures, rowvar=False)
            cov_matrix = np.array(cov)
            p_vector.append(class_pictures.shape[0] / len(Y))
            
        else:
            mu_matrix = np.vstack((mu_matrix, np.mean(class_pictures, axis=0)))
            cov = np.cov(class_pictures, rowvar=False)
            cov_matrix = np.vstack((cov_matrix, np.array(cov)))
            p_vector.append((class_pictures.shape[0] / len(Y)))
            
    return mu_matrix, cov_matrix.reshape(10, dim**2 * 3, dim**2 * 3), p_vector
        

def cifar_10_classifier_bayes(x, mu , sigma, p):
    class_probabilities = []

    for i in range(10):
        # Useing logpdf to avoid overflowing and underflowing
        class_probabilities.append(multivariate_normal.logpdf(x, mu[i], sigma[i]) * p[i])
    return np.argmax(class_probabilities)

def cifar10_2x2_color(X, dim=1):
    scaled_pics = np.array([])
    # Scales and adds the first picture
    scaled_pics = np.array(resize(X[0], (dim, dim, 3), preserve_range=True).reshape(1,dim**2 * 3))
    counter = 0
    for i in range(1, X.shape[0]):
        counter += 1
        scaled_pic = resize(X[i], (dim, dim, 3), preserve_range=True).reshape(1,dim**2 * 3)
        scaled_pics = np.vstack((scaled_pics, scaled_pic))
    return scaled_pics

# Script for Naive Bayesian classifier (Task 1)
X_tr_1x1 = cifar10_color(X_tr)
X_test_1x1 = cifar10_color(X_test)
mu_matrix, std_matrix, p_vector = cifar_10_naivebayes_learn(X_tr_1x1, Y_tr)

predicted_classes = []
print("Running Naive Bayesian classifier")
for pic in X_test_1x1:
    predicted_classes.append(cifar10_classifier_naivebayes(pic, mu_matrix, std_matrix, p_vector))
print(f"Accuracy for naive bayesian (Good): {class_acc(predicted_classes, Y_test)}\n")

# Script for Bayesian classifer (Task 2)
mu_matrix, cov_matrix, p_vector = cifar_10_bayes_learn(X_tr_1x1, Y_tr)
pred_classes = []
print("Running Bayes classifier")
for pic in X_test_1x1:
    pred_classes.append(cifar_10_classifier_bayes(pic, mu_matrix, cov_matrix, p_vector))
print(f"Accuracy for bayes classifier {class_acc(pred_classes, Y_test)}\n")

print("We can see that bayes classifer was slightly better than naive classifier, because we assumed that the rgb-colors are not independent.\n") 

# Script for Task 3 (Best classifier)
pred_classes = []
dim_list = []
acc_list = []
dim = 1
LAST_SIZE = 10
fig, axes = plt.subplots()
axes.set_xlim([0, LAST_SIZE + 1])
axes.set_xlabel("Size (nxn)")
axes.set_ylim([0, 1])
axes.set_ylabel("Accuracy")
axes.set_title("Accuracies for different picture sizes")

while dim <= LAST_SIZE:
    print(f"Running Bayes classifier for {dim}x{dim} picture.")
    X_tr_scaled = cifar10_2x2_color(X_tr, dim=dim)
    X_test_scaled = cifar10_2x2_color(X_test, dim=dim)
    mu_matrix, cov_matrix, p_vector = cifar_10_bayes_learn(X_tr_scaled, Y_tr, dim=dim)
    pred_classes = []
    for pic in X_test_scaled:
        pred_classes.append(cifar_10_classifier_bayes(pic, mu_matrix, cov_matrix, p_vector))
    
    acc = class_acc(pred_classes, Y_test)
    print(f"Accuracy for {dim}x{dim}: {acc}\n")
    plt.plot(dim, acc, "rx")
    dim_list.append(dim)
    acc_list.append(acc)
    dim += 1

plt.plot(dim_list, acc_list, "k-")
print("No need to calculate more, because the accuracy seems to go down")
plt.show()

