# 中文文本分类
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve,train_test_split,GridSearchCV
#Adaboost分类器
from sklearn.ensemble import AdaBoostClassifier
#贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#管道
from sklearn.pipeline import Pipeline
#SVM分类器
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def cut_words(text):
    """
    :param file_path:txt文件路径
    :return:用空格分词的字符串
    """
    text_with_spaces = ''
    textcut = jieba.cut(text)

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    for word in textcut:
        if is_all_chinese(word) == True:
            text_with_spaces += word + ' '
        else:
            pass
    return text_with_spaces


def loadfile(file_list,label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    word_list = []
    labels_list = []
    for file in file_list:
        word_list.append(cut_words(file))
        labels_list.append(label)
    return word_list,labels_list


df1 = pd.read_excel('普通网友325.xlsx')
df2 = pd.read_excel('专业媒体325.xlsx')

data = pd.concat([df1,df2],axis=0)

data1 = data.iloc[0:450]
data2 = data.iloc[450:]

data_label1 = data1[data1['label'] == 1]
data_label2 = data1[data1['label'] == 0]


data_label3 = data2[data2['label'] == 1]
data_label4 = data2[data2['label'] == 0]
# 训练数据
train_words_list1, train_labels1 = loadfile(data_label1['words'], '非负')
train_words_list2, train_labels2 = loadfile(data_label2['words'], '负面')


train_words_list = train_words_list1 + train_words_list2
train_labels = train_labels1 + train_labels2

# 测试数据
test_words_list1, test_labels1 = loadfile(data_label3['words'], '非负')
test_words_list2, test_labels2 = loadfile(data_label4['words'], '负面')

test_words_list = test_words_list1 + test_words_list2
test_labels = test_labels1 + test_labels2

stop_words = open('stopwords_cn.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
# 上面fit过了，这里transform
test_features = tf.transform(test_words_list)



# 构造各种分类器
classifiers = [
    SVC(random_state=10),
    KNeighborsClassifier(metric='minkowski'),
    AdaBoostClassifier(random_state=1)
]

# 分类器名称
classifier_names = [
    'svc',
    'kneighborsclassifier',
    'Adaboostclassifier'
]

# 分类器参数
classifier_param_grid = [
    {'svc__C':[1,5,10], 'svc__gamma':[0.01,0.05,0.001]},
    {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
    {'Adaboostclassifier__n_estimators': [10, 50, 100]}

]


# 对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" %search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" %accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_features, train_labels, test_features, test_labels, model_param_grid, score='accuracy')