from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#构造一个分类器,kernel代表核函数的选择
#linear：线性核函数，线性核函数是在数据线性可分的情况下使用的，运算速度快，效果好，不足在于它不能处理线性不可分的数据
#poly：多项式核函数，多项式核函数可以将数据从低纬度空间映射到高维度空间，但参数比较多，计算量大
#rbf：高斯核函数（默认），高斯核函数同样可以将样本映射到高维空间，但相比于多项式核函数来说所需的参数比较少，通常性能不错，所以是默认使用的核函数
#sigmoid：sigmoid核函数，sigmoid多用于神经网络当中，实现多层神经网络

#参数C代表目标函数的惩罚系数，惩罚系数指的是分错样本时，惩罚的程度，默认情况下为1.0，当C越大的时候，分类器的准确性越高，容错率越低，泛化能力变差
#参数gamma代表核函数的系数，默认为样本征树的倒数，即gamma = 1 /n_features

# model = svm.SVC(kernel='rbf',C=1.0,gamma='auto')

#加载数据集
data = pd.read_csv('./data.csv')

#数据探索
pd.set_option('display.max_columns',None)
print(data.columns)
print(data.head(5))
print(data.describe())

#将特征字段分成3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

#数据清洗
#ID列没有用，删除该列
data.drop('id',axis=1,inplace=True)
#将B良性替换成为0，M恶性替换为1
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label='Count')
plt.show()
#用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()

#特征选择
features_remain = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean']
#抽取30%的数据作为测试集，其余为训练集
train,test = train_test_split(data,test_size=0.3)
#抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_Y = test['diagnosis']

#采用z—score规范化数据，保证每个特征纬度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

#创建SVM分类器
model = svm.SVC()
model.fit(train_X,train_y)
#用测试集做预测
prediction = model.predict(test_X)
print('准确率：',metrics.accuracy_score(prediction,test_Y))


# 创建SVM分类器
model1 = svm.LinearSVC()
# 用训练集做训练
model1.fit(train_X,train_y)
# 用测试集做预测
prediction1=model1.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction1,test_Y))