from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


#获取数据源
x_data = pd.read_csv('分类预测.csv').loc[:,['喜欢吃萝卜','喜欢吃鱼','喜欢捉耗子','喜欢啃骨头','短尾巴','长耳朵']]
#数据类别
y_data = pd.read_csv('分类预测.csv').loc[:,['分类']]
#划分数据集
train_x,test_x,train_y,test_y = train_test_split(x_data,y_data,test_size=0.25)
#对数据类型进行转换
dvec = DictVectorizer(sparse=False)
#转换成数据的格式
train_features = dvec.fit_transform(train_x.to_dict(orient='record'))
test_features = dvec.transform(test_x.to_dict(orient='record'))

#画混淆矩阵
def plot_confusion_matrix(cm,classes, cmap = plt.cm.Blues):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title('分类混淆矩阵准确率100.00%')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)

    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.show()

def knn_class():
    # 创建一个KNN分类器
    knn = KNeighborsClassifier()
    knn.fit(train_features,train_y)
    predict_y = knn.predict(test_features)
    # 评估报告
    target_names = ['兔子', '狗', '猫']
    classification_report1 = classification_report(test_y, predict_y, target_names=target_names)
    print(classification_report1)
    # 混淆矩阵
    confusion = confusion_matrix(test_y, predict_y)
    print(confusion)
    # 混淆矩阵图
    plot_confusion_matrix(confusion,classes=target_names)


# #创建逻辑回归模型分类
def log_class():
    clf = LogisticRegression()
    clf.fit(train_features,train_y)
    predict_y = clf.predict(test_features)
    # 评估报告
    target_names = ['兔子', '狗', '猫']
    classification_report1 = classification_report(test_y, predict_y, target_names=target_names)
    print(classification_report1)
    # 混淆矩阵
    confusion = confusion_matrix(test_y, predict_y)
    print(confusion)
    # 混淆矩阵图
    plot_confusion_matrix(confusion, classes=target_names)


# #创建CART决策树分类器
def cart_class():
    dtc = DecisionTreeClassifier()
    dtc.fit(train_features,train_y)
    predict_y = dtc.predict(test_features)
    # 评估报告
    target_names = ['兔子', '狗', '猫']
    classification_report1 = classification_report(test_y, predict_y, target_names=target_names)
    print(classification_report1)
    # 混淆矩阵
    confusion = confusion_matrix(test_y, predict_y)
    print(confusion)
    # 混淆矩阵图
    plot_confusion_matrix(confusion, classes=target_names)


#随机森林
def random_class():
    rf = RandomForestClassifier()
    parameters = {'n_estimators': range(1,11)}
    clf = GridSearchCV(estimator=rf, param_grid=parameters)
    clf.fit(train_features,train_y)
    predict_y = clf.predict(test_features)
    # 评估报告
    target_names = ['兔子', '狗', '猫']
    classification_report1 = classification_report(test_y, predict_y, target_names=target_names)
    print(classification_report1)
    # 混淆矩阵
    confusion = confusion_matrix(test_y, predict_y)
    print(confusion)
    # 混淆矩阵图
    plot_confusion_matrix(confusion, classes=target_names)


# #创建Naive Bayes分类器
def mnb_class():
    mnb = MultinomialNB()
    mnb.fit(train_features,train_y)
    predict_y = mnb.predict(test_features)
    # 评估报告
    target_names = ['兔子', '狗', '猫']
    classification_report1 = classification_report(test_y, predict_y, target_names=target_names)
    print(classification_report1)
    # 混淆矩阵
    confusion = confusion_matrix(test_y, predict_y)
    print(confusion)
    # 混淆矩阵图
    plot_confusion_matrix(confusion, classes=target_names)

if __name__ == '__main__':
    # KNN
    knn_class()
    #逻辑回归
    log_class()
    #CART决策树
    cart_class()
    # 随机森林
    random_class()
    #朴素贝叶斯
    mnb_class()