import pandas as pd
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import random

# #导入数据集
# df = pd.read_csv('mydata1.csv')
# df1 = pd.read_csv('mydata2.csv')
# df2 = pd.read_csv('mydata3.csv')

#电压
voltage = []
#湿度
humidity = []
#温度
temperature = []
#重量
weight = []

for i in range(1,10000):
    v = random.randint(3,9)
    voltage.append(v)

for i in range(1,10000):
    h = random.randint(40,90)
    humidity.append(h)

for i in range(1,10000):
    t = random.randint(20,32)
    temperature.append(t)

for i in range(1,10000):
    w = random.randint(300,700)
    weight.append(w)




#创建dataframe
df3 = pd.DataFrame()
df3['voltage'] = voltage
df3['humidity'] = humidity
df3['temperature'] = temperature
df3['weight'] = weight
#采用无监督学习让数据进行聚类
k_means = cluster.KMeans(n_clusters=5)
#传入数据
k_means.fit(df3)
#创建新的一列把聚好的类传入到该列里面
df3['label'] = k_means.labels_[::]

#把上面创建的好的dataframe进行划分数据集
data = df3[['voltage','humidity','temperature']]
#标签
target = df3['label']

#对数据集进行划分为训练集和测试集，划分百分之25给测试集，其余用于训练集
train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.25)
#让数据标准化有利于提高准确率
# 采用z—score规范化数据，保证每个特征纬度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_x)
test_X = ss.transform(test_x)

#创建SVM分类器
model = svm.SVC(kernel='poly',C=10.0,gamma='auto')
model.fit(train_X,train_y)
#用测试集做预测
prediction = model.predict(test_X)
#测试结果如下，到时候你要把数据进行分类也是去查看prediction就是分类好的结果
print(prediction)
#查看它的准确率，准确率为1很好，不过也和训练集有关，数据量太少，导致准确率偏高
print('准确率：',metrics.accuracy_score(prediction,test_y))
