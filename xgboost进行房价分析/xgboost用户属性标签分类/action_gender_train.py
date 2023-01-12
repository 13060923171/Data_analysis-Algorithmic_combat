from numpy import loadtxt
#xgboost分类
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

dataset = loadtxt('data/action_gender_train_data.csv', delimiter=",")

X = dataset[:, 0:4]
Y = dataset[:, 4]
#防止过拟合的出现
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
model.fit(X, Y)
model.save_model("action_gender.model")
print("params:", model.get_params())
# 显示特征的重要程度
plot_importance(model)
pyplot.show()

inputSet = loadtxt('data/action_gender_input_data.csv', delimiter=",")
xinput = inputSet[:, 0:4]
predictResult = model.predict(xinput)
print(len(predictResult))
print(predictResult)
