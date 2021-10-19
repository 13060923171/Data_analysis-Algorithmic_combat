import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


fig1=plt.figure()
plt.ylabel("S$ MILLION")
plt.xlabel("FINANCIAL YEAR")
plt.title("Government Expenditure on Health")
bar_chart=plt.bar(["2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"],[2009.7,2283.2,2814.1,3745.8,3856.7,4091.5,4837.3,5938.1,7223.1,8639.9,9307,9764.3,10122.7],color="orange")

fig2=plt.figure()
plt.ylabel("Number of Providers")
plt.xlabel("YEAR")
plt.title("Number of Centre-based Care Facilities")
plt.plot([2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],[32,33,33,35,34,35,44,57,68,81,88,102,123])

fig3=plt.figure()
plt.ylabel("Number of Providers")
plt.xlabel("S$ MILLION")
plt.title("Number of Providers vs Government Expenditure on Health")
x = np.array([float(2009.7),float(2283.2),float(2814.1),float(3745.8),float(3856.7),float(4091.5),float(4837.3),float(5938.1),float(7223.1),float(8639.9),float(9307),float(9764.3),float(10122.7)])
y = np.array([float(32),float(33),float(33),float(35),float(34),float(35),float(44),float(57),float(68),float(81),float(88),float(102),float(123)])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(x,y)
m,c = np.polyfit(x,y,1)
print(m,c)
y2 = m*x+c
plt.text(1,1,'线性回归:y2 = 0.01005644*x+1.11178275400 \nr2 = 1-SSE/SST =0.9841081971176419',bbox={'facecolor':'cyan','alpha':0.5,'pad':1})  #添加文本框
plt.plot(x,y2,"r")
plt.show()

# 模型训练
model = sm.OLS(y, x).fit()
# 提取R方的回归系数
a = model.rsquared
print(a)
