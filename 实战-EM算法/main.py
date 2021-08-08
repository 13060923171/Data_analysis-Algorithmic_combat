from sklearn.mixture import GaussianMixture
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
#创建GMM聚类
#首先n_components既高斯混合模型的个数，也就是我们要聚类的个数，默认值为1，如果不指定n_components，最终的聚类结果都会为同一个值
#covariance_type代表协方差类型，一个高斯混合模型的分布是由均值向量和协方差矩阵决定的，协方差的类型也代表了不同的高斯混合模型的特征
#covariance_type='full'，代表完全协方差，也就是元素都不为0
#covariance_type='tied'，代表相同的完全协方差
#covariance_type='diag',代表对角协方差，也就是对角不为0，其余为0
#covariance_type='spherical'，代表球面协方差，非对角为0，对角完全相同，呈现球面的特性
#max_iter代表最大迭代次数，EM算法是由E步和M步迭代求得最终的模型参数，这里可以指定最大迭代次数，默认值为100
gmm1 = GaussianMixture(n_components=1,covariance_type='full',max_iter=100)

#数据加载.避免中文乱码问题
data_ori = pd.read_csv('heros.csv',encoding='gb18030')
features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',u'初始法力',u'最高物攻',u'物攻成长',u'初始物攻',u'最大物防',u'物防成长',u'初始物防', u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features]

#对英雄属性之间的关系进行可视化分析
#设置plt正确显示中文
#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
#用热力图呈现features_mean字段之间的相关性
corr = data[features].corr()
plt.figure(figsize=(14,14))
#annot=True显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()

#相关性最大的属性保留一个，因此可以对属性进行降维
features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features_remain]
data[u'最大攻速'] = data[u'最大攻速'].apply(lambda x:float(x.strip('%')) / 100)
data[u'攻击范围'] = data[u'攻击范围'].map({'远程':1,'近战':0})
#采用Z-score规范化数据，保证每个特征维度的数值均值为0，方差为1
ss = StandardScaler()
data = ss.fit_transform(data)
#构造GMM聚类
gmm = GaussianMixture(n_components=30,covariance_type='full')
gmm.fit(data)
#训练数据
prediction = gmm.predict(data)
print(prediction)
#将分组结果输出到CSV中
data_ori.insert(0,'分组',prediction)
data_ori.to_csv('heros1.csv',index=False,sep=',')
#计算聚类结果的指标,指标分数越高，代表聚类结果越好，也就是相同类中的差异性小，不同类的差异性大
print(calinski_harabasz_score(data,prediction))