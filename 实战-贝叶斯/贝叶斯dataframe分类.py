# 中文文本分类
import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
warnings.filterwarnings('ignore')

def cut_words(text):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    words_list = []
    labels_list = []
    for file in file_dir:
        words_list.append(cut_words(file))
        labels_list.append(label)
    return words_list, labels_list


df = pd.read_csv('new_离婚冷静期.csv').iloc[:500]
df1 = df['内容'][df['分类'] == '其他']
df2 = df['内容'][df['分类'] == '家暴']
df3 = df['内容'][df['分类'] == '性别对立']
df4 = df['内容'][df['分类'] == '女性不平等']

# 训练数据
train_words_list1, train_labels1 = loadfile(df1, '其他')
train_words_list2, train_labels2 = loadfile(df2, '家暴')
train_words_list3, train_labels3 = loadfile(df3, '性别对立')
train_words_list4, train_labels4 = loadfile(df4, '女性不平等')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4
#
# # 测试数据

df11 = pd.read_csv('new_离婚冷静期.csv')
# df5 = df11['内容'][df11['分类'] == '其他']
# df6 = df11['内容'][df11['分类'] == '家暴']
# df7 = df11['内容'][df11['分类'] == '性别对立']
# df8 = df11['内容'][df11['分类'] == '女性不平等']

test_words_list1, test_labels1 = loadfile(df11['内容'], '其他')
# test_words_list2, test_labels2 = loadfile(df6, '家暴')
# test_words_list3, test_labels3 = loadfile(df7, '性别对立')
# test_words_list4, test_labels4 = loadfile(df8, '女性不平等')

test_words_list = test_words_list1
test_labels = test_labels1

stop_words = open('stopwords_cn.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
# 上面fit过了，这里transform
test_features = tf.transform(test_words_list)

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

result = pd.concat((df11, pd.DataFrame(predicted_labels)), axis=1)
result.rename({0: '分类结果'}, axis=1, inplace=True)
result.to_csv('new_class.csv',encoding="utf-8-sig")
# # 计算准确率
# print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))



