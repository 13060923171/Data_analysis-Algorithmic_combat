from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
#KNN做分类的话是引用KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
#KNN做回归的话是引用KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


#创建一个分类器
#n_neighbors既表示KNN中的K值，代表的是邻居的数量，K值如果比较小，会造成过拟合。如果K值比较大，无法将无知物体分类出来，一般我们使用默认值是5
#weights=uniform代表所有邻居的权重相同
#weights=distance，代表权重是距离的倒数，既与距离成反比
#自定义函数，你可以自定义不同距离所对应的权重。大部分情况下不需要自己定义函数
#algorithm用来规定计算邻居的方法
#algorithm=auto，根据数据的情况自动选择适合的算法，默认情况选择auto
#algorithm=kd_tree也叫KD树，是多维空间的数据结构，方便对关键数据进行检索，不过KD树适用于维度少的情况，一般维数不超过20，如果超过20效率会下降
#algorithm=ball_tree也叫球树，它和KD树一样都是多维空间的数据结果，不同于KD树，球树更适用于维度大的情况
#algorithm=brute也叫暴力搜索，它和KD树不同的地方在于采用的是线性扫描，而不是通过构造树结构进行快速检索，当训练集大的时候，效率比较低
#leaf_size代表构造KD树或者球树的叶子树，默认为30，leaf_size会影响到树的构造和搜索速度
clf = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',leaf_size=30)

#首先加载数据
digits = load_digits()
data = digits.data
#数据探索
print(data.shape)
print(digits.images[0])
print(digits.target[0])
#将图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

#分割数据，将25%的数据作为测试集，其余作为训练集
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25)
#采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#创建一个KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y = knn.predict(test_ss_x)
print('KNN准确率:%.4lf' % accuracy_score(test_y,predict_y))


#创建一个SVM分类器
svm = SVC()
svm.fit(train_ss_x,train_y)
predict_y1= svm.predict(test_ss_x)
print('SVM准确率:%0.4lf' % accuracy_score(test_y,predict_y1))

#采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)
#创建Naive Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x,train_y)
predict_y2 = mnb.predict(test_mm_x)
print('NB准确率:%0.4lf' % accuracy_score(test_y,predict_y2))

#创建CART决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_x,train_y)
predict_y3 = dtc.predict(test_mm_x)
print("CART决策树准确率: %.4lf" % accuracy_score(test_y,predict_y3))