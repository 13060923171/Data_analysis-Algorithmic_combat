import pandas as pd
#分类
from sklearn.model_selection import train_test_split
#数据规范化
from sklearn import preprocessing
#svm
from sklearn import svm
#测试分数
from sklearn import metrics
#随机森林
from sklearn.ensemble import RandomForestClassifier

#调用数据
data = pd.read_csv('banbook_data.csv').loc[:,['data1','data2','data3','data4']]
data = data.values
target = pd.read_csv('banbook_data.csv').loc[:,['labels']]
target = target.values
#切割数据，抽取百分之30作为测试集，其余作为训练集
train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.3)

#采用z—score规范化数据，保证每个特征纬度的数据均值为0，方差为1
ss = preprocessing.StandardScaler()
#transform则是根据对之前部分训练数据进行fit的整体指标，对测试数据集使用同样的均值，方差，最大，最小值等指标进行转换transform（testData），从而保证train，test处理方式相同
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
list.sort()


def get_svm():
    #创建SVM分类器
    model = svm.LinearSVC()
    model.fit(train_ss_x,train_y)
    #用测试集做预测
    prediction = model.predict(test_ss_x)
    print('svm准确率：',metrics.accuracy_score(prediction,test_y))


def get_random_forest():
    # 创建随机森林分类器 n_estimators随机森林中决策树的个数，n_jobs表示拟合和预测时CPU的核数
    forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    #对数据进行分类
    forest.fit(train_ss_x,train_y)
    # 用测试集做预测
    prediction = forest.predict(test_ss_x)
    print('forest准确率：', metrics.accuracy_score(prediction, test_y))

if __name__ == '__main__':
    get_svm()
    get_random_forest()