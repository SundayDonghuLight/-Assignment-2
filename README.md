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
mnist = fetch_mldata('MNIST original')</pre></code>
  </li>
  <li>
    產生訓練資料與測試資料：
    <p>mnist總共有70000筆data，每筆皆為28*28像素的手寫數字灰度圖，而target則是對應的數字。</p>
    <p>由於原始資料有照0~9的順序排列，所以不能直接將前60000筆當作訓練資料，後10000萬筆當測資，</p>
    <p>這邊靠<code>random.shuffle</code>的應用從這70000筆資料中隨機挑60000筆作訓練資料，剩下10000萬筆當測試資料。</p>
    <p>將訓練用的data與target分別存入X_train與y_train中，測試資料同樣分別存入X_test與y_test中。</p>
<pre><code>index = np.arange(len(mnist.data))
random.shuffle(index)
train_index = index[0:60000]
test_index = index[60000:70000]
X_train, y_train = mnist.data[train_index], mnist.target[train_index]
X_test, y_test = mnist.data[test_index], mnist.target[test_index]</pre></code>
  </li>
  <li>
    Dimension Reduction：
    <p>使用sklearn.decomposition.PCA套件進行降維，參數<code>n_components</code>設為30，即將數據降至30維。</p>
    <p>經測試在降到30維左右時有著最佳的辨識率，詳細數據可參考根目錄下PCA - reduced dimensions compare.ipynb檔案。</p>
    <p>維度化減完在進行辨識器的訓練前我們先對資料進行歸一化(normalization)的動作，</p>
    <p>把每一筆input data都除上255使其值落在0~1之間。</p>
<pre><code>pca = PCA(n_components=30)
newX_train = pca.fit_transform(X_train)
newX_test = pca.transform(X_test)
newX_train = newX_train/255
newX_test = newX_test/255</pre></code>
  </li>
  <li>
    訓練與辨識：
    <p>採用Support Vector Machine作為我們的辨識器，用sklearn通用的<code>clf.fit</code>函數來進行模型的訓練。</p>
    <p>最後的指令會產生一條與資料量等長的陣列，如果該項的預測結果與其實際label相同為1，不同則為2，</p>
    <p>將<code>np.mean</code>function作用在這個陣列後即可得到辨識率。</p>
<pre><code>clf = SVC()
clf.fit(newX_train, y_train)
print("訓練資料辨識率:",np.mean(clf.predict(newX_train) == y_train))
print("測試資料辨識率:",np.mean(clf.predict(newX_test) == y_test))
</pre></code>
  </li>
</ol>

### 運行結果
<p>　　PCA：降至30維 (n_components=30)</p>
<p>　　辨識器：SVC (Support Vector Classification)</p>
<p>　　訓練資料辨識率：99.10%</p>
<p>　　測試資料辨識率：98.54%</p>

### 化減至不同維度的比較
<p>　　在"PCA - reduced dimensions compare.ipynb"檔案中有做比較</p>
<p>　　可以看到解釋率在95~97%時確實有不錯的辨識率，不過以這個例子來說，最佳的範圍是在降至更低後的30維左右。</p>
<p>　　而在絕大多數的範圍內，使用PCA都能保持不錯的辨識率，即便降到了10維也都還有95%左右的辨識率，</p>
<p>　　再更低的5或1時才有比較顯著的負面影響。</p>

### 不同辨識器間的比較
<p>　　可參考"Different classifiers comparison.ipynb"檔案</p>
<p>　　使用sklearn套件，目前所學的方法中辨識效果最佳的為SVC辨識器，</p>
<p>　　而類神經網路也有著差不多的辨識率與更快一些的速度，其他幾種線性辨識器效果就差上了一截。</p>
<p>　　比較讓我感到意外的是KNeighborsClassifier居然發揮了這麼高的成效，雖然花了較久的時間，但辨識率直逼SVC，</p>
<p>　　一開始還因為其單純的概念有點小看他，現在知道在進行辨識時，KNeighbors還是相當值得一試的。</p>
<p>　　再來還可以看到一些有趣的發現，像是DecisionTree如果用原訓練資料下去跑辨識率必然會是100%這點，</p>
<p>　　和雖然1棵樹只有84%，但用RandomForest就有92%的辨識率了，且可以注意到訓練資料的辨識率不一定會再是100%。</p>
<p>　　最後幾個Ensemble Methods有點小失誤，沒注意看就直接用了預設，而Bagging的預設就是DecisionTree，</p>
<p>　　所以其實跟使用RandomForest是一樣的。</p>

### 心得
<p>因為寫這份作業的同時也有著期末人臉辨識的專案要進行，所以在這兩邊交錯的嘗試下真的覺得自己比原本進步了很多，一開始作業一時還完全用不好的sklearn套件，現在可以說是相當習慣了。再來3個這次作業中讓我覺得特別意義重大的收穫就是對標準化(normalization)，或著說是歸一化的意識，一開始在做這份手寫字辨識時就是缺了這一個步驟，結果SVC辨識的一蹋糊塗且每次訓練還都要用2個小時，怎麼調參數都不見有改善的情形，真的困擾了很久甚至一度放棄了使用SVC的念頭。</p>
<p>直到後來完成後還有時間才又回來開始查找導致SVC辨識率低下的原因，最後在嘗試了normalization後獲得了驚人的改善，一次給了這麼強的印象我以後想要再忘掉應該是挺難了吧哈哈。還有個意外的收穫是連正則化(Regularization)的觀念也一起弄清楚了，因為正則化、標準化這兩個的中文差不多，看了實在分不清楚就閱讀了不少的資料，且剛開始搜尋overfitting時，跳出來的也幾乎都是l1,l2正則化相關的文章。</p>
<p>再來則分別是PCA不同components下的比較和各種辨識器間的比較，如果沒有進行這次PCA的實驗，恐怕還真不會意識到這個方法是這麼的強大，即便降至10維了仍保有相當水準的辨識率，這樣以後使用時可以比較放心的嘗試降到更低維，而不會受制於保守的思想放棄這些可能。</p>
<p>而進行辨識器間的比較時不僅是更熟練了sklearn套件的使用，同時也有了對這整學期所學的知識進行總複習的感覺，在使用某種辨識器時想著其運作的方法，為什麼會得到這些結果等等，在實做中體現真的可以說是最棒的複習了，也謝謝老師這一學的教導和助教們的辛勞，讓我可以在這麼好的環境下學習這有趣的一門課。</p>
<p>　　</p>
<p>　　</p>
