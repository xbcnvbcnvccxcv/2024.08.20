import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from  sklearn.model_selection import  train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import scipy


header= ['CRIM','ZN', 'INCUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv("./data/3.housing.csv", delim_whitespace= True, names= header)

array = data.values
X = data['RM'].values.reshape(-1, 1) # [행,열]
Y = data['MEDV'].values
X = X.reshape(-1,1)
# 데이터 분할(Train, Test)

plt.clf()
plt.scatter(X, Y, color='blue', label='Actual values')
plt.legend()
plt.xlabel("Crime Rate")
plt.ylabel("House Price")
plt.show()

#모델 선택 및 학습
# #방정식 찾게하기
# model = LinearRegression()
# model.fit(X_train, Y_train)
# model.coef_
# model.intercept_
#
# #모델 예측
# y_pred = model.predict(X_test) #근속연속이 있을 때 연봉을 예측해봐
# # print(y_pred[:,1])
# error = mean_absolute_error(y_pred, Y_test)
# print(error)

plt.clf()
plt.scatter(X_test[:100], Y_test[:100], color='blue', label='Actual values')
plt.plot(range(len(y_pred[:100])), y_pred[:100], color='red', label='Predicted values', marker='o')
plt.legend()
plt.xlabel("Weight(kg)")
plt.ylabel("Height(cm)")
# plt.savefig("./WW/")
# plt.show()
