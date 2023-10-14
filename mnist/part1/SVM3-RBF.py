import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()

"""n_components = 10
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

svm_model = svm.SVC(kernel='poly',degree=3,random_state=0)
Model = svm_model.fit(train_cube,train_y)

y_p = Model.predict(test_cube)
test_error_SVM3 = compute_test_error_svm(test_y, y_p)
print("Error on the test set:", test_error_SVM3)"""
from sklearn.decomposition import PCA
from sklearn.svm import SVC
pca = PCA(n_components=10, random_state=0)
X_train_pca = pca.fit_transform(train_x)
X_test_pca = pca.transform(test_x)
#train_cube = cubic_features(X_train_pca)
#test_cube = cubic_features(X_test_pca)

#clf = SVC(kernel='poly', random_state=0)
clf = SVC(kernel='rbf', random_state=0)

clf.fit(X_train_pca,train_y)
error_rate = 1 - clf.score(X_test_pca,test_y)

#print('Error rate for 10-dimensional PCA features using poly 3rd SVM: {:.4f}'.format(error_rate))
print('Error rate for 10-dimensional PCA features using RBF SVM: {:.4f}'.format(error_rate))