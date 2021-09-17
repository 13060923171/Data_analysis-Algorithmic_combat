import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm
import seaborn as sns  # visualization

# 整体数据直方图
plt.rcParams['figure.figsize'] = (15,8) 
plt.rcParams['axes.titlesize'] = 'large'

data = pd.read_csv('data/abalone.csv', index_col="Unnamed: 0")
data.head(5)
print(data)

plt.title('The Statistical Figure')
sns.set_style("white")
sns.set_context({"figure.figsize": (10, 8)})
sns.countplot(x=data['Rings'],label='Count', palette="Set3")
plt.show()



#将数据分为训练集与测试集
from sklearn.model_selection import train_test_split
features = data[data.loc[:,data.columns!='Rings'].columns] #提取特征
target = data['Rings'] #提取目标特征
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=33)

features=features.drop("Sample code number",axes=1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_train = ss.fit_transform(x_train)  # fit_transform for train data
x_test = ss.transform(x_test)


from sklearn.linear_model import LogisticRegression #从sklearn中引入逻辑回归

lr = LogisticRegression(solver='lbfgs' ) #进行初始化
lr.fit(x_train, y_train) #进行训练
y_pred = lr.predict(x_test) #进行预测
