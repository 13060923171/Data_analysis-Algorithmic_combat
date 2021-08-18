#用ARMA进行时间序列预测
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.api import qqplot

#创建数据
data = [5922,5308,5546,5975,2706,1751,4111,5592,4725,2313,4542,9685,3672,6789,5858]
data = pd.Series(data)
print(data)
data_index = sm.tsa.datetools.dates_from_range('1901','1915')
print(data_index)
#绘制数据图
data.index = pd.Index(data_index)
data.plot(figsize=(12,8))
plt.show()
#创建ARMA模型
arma = ARMA(data,(7,0)).fit()
# print('AIC: %0.4lf' %arma.aic)
# #模型预测
# predict_y = arma.predict('1915','1920')
# #预测结果绘制
# fig,ax = plt.subplots(figsize=(12,8))
# ax = data.ix['1901':].plot(ax=ax)
# predict_y.plot(ax=ax)
# plt.show()