# Introductory Machine Learning Programming Assignment 2
## Handwritten Digit Recognition (MNIST 手寫數字資料集辨識)

### 程式說明
<ol>
  <li>
    載入套件與資料集: <ul>
    <li>numpy: 能在python中很直覺的實現數學上各種矩陣運算的強大套件</li>
    <li>random: 包含許多與隨機亂數相關指令的套件，會用到其中<code>random.shuffle</code>的洗牌功能</li>
    <li>matplotlib.pyplot: 能進行繪圖，可用來顯示和儲存圖像，使用上與MATLAB相似</li>
    <li>sklearn.decomposition.PCA: <strong>sklearn</strong>的主成分分析套件，可用來對高維度的資料進行降維化簡</li>
    <li>sklearn.svm.SVC: <strong>sklearn</strong>中用來處理分類問題的支持向量機(Support Vector Machine)</li>
    <li>sklearn.datasets.fetch_mldata: <strong>sklearn</strong>中的數據資料包，裡面有要用的MNIST手寫數字資料集</li>
    </ul>
<pre><code>import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')                #載入'MNIST original'資料集存入變數mnist中</pre></code>
  </li>
  <li>
    產生訓練資料與測試資料:
    mnist總共有70000筆data
    由於
<pre><code>index = np.arange(len(mnist.data))                                     #產生與data數相同長度(0~69999)的陣列
random.shuffle(index)                                                  #將該陣列隨機洗牌，打亂順序
train_index = index[0:60000]                                           #前60000筆對應的編號作為訓練資料
test_index = index[60000:70000]                                        #後10000筆對應的編號則作為測試資料
X_train, y_train = mnist.data[train_index], mnist.target[train_index]
X_test, y_test = mnist.data[test_index], mnist.target[test_index]</pre></code>
  </li>
</ol>
