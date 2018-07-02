import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')                                       #載入'MNIST original'手寫字資料集存入變數mnist中

index = np.arange(len(mnist.data))                                           #產生與data數相同長度(0~69999)的陣列
random.shuffle(index)                                                        #將該陣列隨機洗牌，打亂順序
train_index = index[0:60000]                                                 #前60000筆對應的編號作為訓練資料
test_index = index[60000:70000]                                              #後10000筆對應的編號則作為測試資料
X_train, y_train = mnist.data[train_index], mnist.target[train_index]
X_test, y_test = mnist.data[test_index], mnist.target[test_index]

pca = PCA(n_components=30)                                                   #使用PCA套件，設定降至30維
newX_train = pca.fit_transform(X_train)                                      #用訓練資料配適降階用的半正交矩陣並把轉換後結果存至newX_train
newX_test = pca.transform(X_test)                                            #用剛剛找出的矩陣對訓練資料也進行轉換，存入newX_test
newX_train = newX_train/255                                                  #將資料進行歸一化(normalization)
newX_test = newX_test/255                                                    #這邊除255等同於sklearn的MinMaxScaler

clf = SVC()                                                                  #使用SVC(Support Vector Classification)進行辨識
clf.fit(newX_train, y_train)
print("訓練資料辨識率:",np.mean(clf.predict(newX_train) == y_train))          #看對訓練資料進行預測後與其實際label相同的比例作為訓練資料辨識率
print("測試資料辨識率:",np.mean(clf.predict(newX_test) == y_test))            #看對測試資料進行預測後與其實際label相同的比例作為測試資料辨識率
