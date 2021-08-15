#回归
from sklearn.ensemble import AdaBoostRegressor
#分类
#决策树回归
from sklearn.ensemble import AdaBoostClassifier
#KNN回归
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#base_estimate代表弱分类器。
#n_estimators代表算法的最大迭代次数，也是分类器的个数，每一次迭代都会引入一个新的弱分类器来增加原有的分类器的组合能力
#learing_rate代表学习率，取值在0-1之间，默认是1.0，如果学习率小，就需要比较多的迭代次数才能收敛，也就是学习率和迭代次数都是有相关性的
#当要调整learning_rate的时候，也需要调整n_estimators这个参数
#algorithm代表我们要采用哪种boosting算法，一共有两种SAMME和SAMME.R。默认是SAMME.R，这两者的区别在于对弱分类器权重的计算方式不同
#random_state代表随机种子的设置，默认为None，随机种子是用来控制随机模式的，当随机种子取了一个值，也就是确定一种随机规则，其他人取这个值可以得到同样的结果
# Ada = AdaBoostClassifier(base_estimator=None,n_estimators=50,learning_rate=1.0,algorithm='SAMME.R',random_state=None)

#加载数据
data = load_boston()
#分割数据
train_x,test_x,train_y,test_y = train_test_split(data.data,data.target,test_size=0.3)
#使用AdaBoost回归模型
regress = AdaBoostRegressor()
regress.fit(train_x,train_y)
pred_y = regress.predict(test_x)
mse = mean_squared_error(test_y,pred_y)
print('房价预测结果',pred_y)
print('均方误差=',round(mse,2))



#使用决策树做回归模型
dec_regressor = DecisionTreeRegressor()
dec_regressor.fit(train_x,train_y)
pred_y = dec_regressor.predict(test_x)
mse1 = mean_squared_error(test_y,pred_y)
print('决策树均方误差=',round(mse1,2))

#使用KNN回归模型
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x,train_y)
pred_y = knn_regressor.predict(test_x)
mse2 = mean_squared_error(test_y,pred_y)
print('KNN均方误差=',round(mse2,2))

#在相比之下，Adaboost的均方误差更小，也就是结果更优，虽然Adaboost使用了弱分类器，但是通过50个甚至更多的弱分类器组合起来而形成的强分类器，
#在很多情况下结果都优于其他算法，因此Adaboost也是常用的分类和回归算法之一

