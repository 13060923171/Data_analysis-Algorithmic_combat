import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

df = pd.read_excel('new_data.xlsx',sheet_name='Sheet2')

X = df[['标题1','标题2','标题3','标题4','标题5']].values.tolist()
y = df['点击率'].values.tolist()
# 将数据合并为一个列表
data = [' '.join(sentence) for sentence in X]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data).toarray()
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# 初始化特征标准化器
scaler = StandardScaler()

# 特征标准化
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def cs():
        # 定义并训练GBT回归模型
        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        reg.fit(X_train, y_train)
        # 定义参数列表
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        # 网格搜索
        grid_search = GridSearchCV(reg, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print('最优参数：', grid_search.best_params_)

cs()
# 定义并训练GBT回归模型
reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
reg.fit(X_train, y_train)
# 用测试数据进行预测，并计算MSE和R2值
y_pred = reg.predict(X_test)
mse_reg = mean_squared_error(y_test, y_pred)
r2_reg = r2_score(y_test, y_pred)
print('MSE: %.4f' % mse_reg)
print('R2: %.4f' % r2_reg)


# 初始化线性回归模型
lr = LinearRegression()
# 训练模型
lr.fit(X_train, y_train)
# 在测试集上进行预测
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print('MSE: %.4f' % mse_lr)
print('R2: %.4f' % r2_lr)


# 定义多个模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lr = LinearRegression()

# 定义投票回归器
er = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)])

# 进行模型训练和预测
er.fit(X_train, y_train)
y_pred = er.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE: %.4f' % mse)
print('R2: %.4f' % r2)

x_data1 = ['gbtd_mse','lr_mse','er_mse']
x_data2 = ['gbtd_r2','lr_r2','er_r2']
y_data1 = [mse_reg,mse_lr,mse]
y_data2 = [r2_reg,r2_lr,r2]
df1 = pd.DataFrame()
df1['MSE_index'] = x_data1
df1['MSE_values'] = y_data1
df1['R2_index'] = x_data2
df1['R2_values'] = y_data2
df1.to_csv('result.csv',encoding='utf-8-sig')