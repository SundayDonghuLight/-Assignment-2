# Introductory Machine Learning Programming Assignment 2
## Handwritten Digit Recognition (MNIST 手寫數字資料集辨識)

### 程式說明
<ol>
  <li>
    載入套件與資料集： <ul>
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
    產生訓練資料與測試資料：
    <p>mnist總共有70000筆data，每筆皆為28*28像素的手寫數字灰度圖，而target則是對應的數字。</p>
    <p>由於原始資料有照0~9的順序排列，所以不能直接將前60000筆當作訓練資料，後10000萬筆當測資，</p>
    <p>這邊靠<code>random.shuffle</code>的應用從這70000筆資料中隨機挑60000筆作訓練資料，剩下10000萬筆當測試資料。</p>
    <p>將訓練用的data與target分別存入X_train與y_train中，測試資料同樣分別存入X_test與y_test中。</p>
<pre><code>index = np.arange(len(mnist.data))                                     #產生與data數相同長度(0~69999)的陣列
random.shuffle(index)                                                  #將該陣列隨機洗牌，打亂順序
train_index = index[0:60000]                                           #前60000筆對應的編號作為訓練資料
test_index = index[60000:70000]                                        #後10000筆對應的編號則作為測試資料
X_train, y_train = mnist.data[train_index], mnist.target[train_index]
X_test, y_test = mnist.data[test_index], mnist.target[test_index]</pre></code>
  </li>
  <li>
    Dimension Reduction：
    <p>使用sklearn.decomposition.PCA套件進行降維，參數<code>n_components</code>設為30，即將數據降至30維。</p>
    <p>經測試在降到30維左右時有著最佳的辨識率，詳細數據可參考根目錄下PCA - reduced dimensions compare.ipynb檔案。</p>
    <p>維度化減完在進行辨識器的訓練前我們先對資料進行歸一化(normalization)的動作，把每一筆input data都除上255使其值落在0~1之間</p>
<pre><code>pca = PCA(n_components=30)                  #使用PCA套件，設定降至30維
newX_train = pca.fit_transform(X_train)     #用訓練資料配適降階用的半正交矩陣並把轉換後結果存至newX_train
newX_test = pca.transform(X_test)           #用剛剛找出的矩陣對訓練資料也進行轉換，存入newX_test
newX_train = newX_train/255                 #將資料進行歸一化(normalization)
newX_test = newX_test/255</pre></code>
  </li>
</ol>


<p></p>
<p></p>
