from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score



def get_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # 数据探索
    # print(train_data.info())
    # print('-'*30)
    # print(train_data.describe())
    # print('-'*30)
    # print(train_data.describe(include=['O']))
    # print('-'*30)
    # print(train_data.head())
    # print('-'*30)
    # print(train_data.tail())

    #使用平均年龄来填充年龄中的nan值
    train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
    #使用票价的均值填充票价中的nan值
    train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

    #查看某个字段的取值
    # print(train_data['Embarked'].value_counts())
    #使用登录最多的港口来填充登录港口的nan值
    train_data['Embarked'].fillna('S',inplace=True)
    test_data['Embarked'].fillna('S',inplace=True)
    #特征选择
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]
    #在特征值不都是数值类型的时候，可以使用DictVectorizer类进行转化
    dvec=DictVectorizer(sparse=False)
    #在这里有注意的一点就是fit_transform和transform这二者的区别，二者的功能都是对数据进行某种统一的处理，（比如标准化~N（0,1），将数据缩放（映射）到某个固定区间，归一化，正则化等）
    #fit_transform（trainData对部分训练数据先拟合fit，找到部分训练数据的整体指标，如均值，方差，最大值最小值等等，然后对训练数据进行转换transform，从而实现数据的标准化，归一化等等）
    #transform则是根据对之前部分训练数据进行fit的整体指标，对测试数据集使用同样的均值，方差，最大，最小值等指标进行转换transform（testData），从而保证train，test处理方式相同
    train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
    test_features = dvec.transform(test_features.to_dict(orient='record'))
    # print(dvec.feature_names_)

    return train_features,test_features,train_labels

if __name__ == '__main__':
    #获取数据源
    train_features, test_features, train_labels = get_data()
    #构造ID3决策树
    clf = DecisionTreeClassifier(criterion='entropy')
    #决策树训练
    clf.fit(train_features,train_labels)
    #决策树预测
    pred_label = clf.predict(test_features)
    acc_decision_tree = round(clf.score(train_features, train_labels), 6)
    #正常的准确率评估
    print(u'score 准确率为 %.4lf' % acc_decision_tree)
    #采用K折交叉验证的方式，在不知道测试集的实际结果的时候，要使用K折交叉验证才能知道模型的准确率
    print(u'cross_val_score准确率 %.4lf' % np.mean(cross_val_score(clf,train_features,train_labels)))
