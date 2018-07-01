import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original') 

index = np.arange(len(mnist.data))
random.shuffle(index)
train_index = index[0:60000]
test_index = index[60000:70000]
X_train, y_train = mnist.data[train_index], mnist.target[train_index]
X_test, y_test = mnist.data[test_index], mnist.target[test_index]

pca = PCA(n_components=60)
newX_train = pca.fit_transform(X_train)
newX_test = pca.transform(X_test)
newX_train = newX_train/255
newX_test = newX_test/255

clf = SVC()
clf.fit(newX_train, y_train)
print(np.mean(clf.predict(newX_train) == y_train))
print(np.mean(clf.predict(newX_test) == y_test))
