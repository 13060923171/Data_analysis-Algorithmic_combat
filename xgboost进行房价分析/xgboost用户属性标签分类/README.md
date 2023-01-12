性别预测

以用户中心用户自填的性别作为训练集。用户自填的性别和问卷调研的性别重合度较高。

根据用户对商品二级类目的行为预测用户性别

构造特征：


label: 0表示男性，1表示女性


特征：

用户所操作的二级类目id、三级类目id、行为类型、行为次数

训练集：mining/xgboost/data/action_gender_train_data.csv

训练数据中的列名：cat2_id,cat3_id,action,action_cnt,user_gender

模型训练运行action_gender_train.py


输入的性别未知的数据：action_gender_train_data.csv

前四列也为格式相同的特征：

cat2_id,cat3_id,action,action_cnt,user_id

输入数据的获取：user_gender=2表示性别未知的用户

