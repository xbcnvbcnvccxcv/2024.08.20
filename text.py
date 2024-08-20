import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from  sklearn.model_selection import  train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import scipy



data = pd.read_csv('./data/3.housing.csv', index_col = 0)
data['Height(cm)']
print(data['Height(cm)'])

data['Weight(kg)']
print(data['Weight(kg)'])


array = data.values
X = array[:,0] # [행,열]
Y = array[:,1]
X = X.reshape(-1,1)
# 데이터 분할(Train, Test)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
print((X_test.shape, X_test.shape, Y_train.shape, Y_test.shape))

#모델 선택 및 학습
#방정식 찾게하기
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_

#모델 예측
y_pred = model.predict(X_test) #근속연속이 있을 때 연봉을 예측해봐
# print(y_pred[:,1])
error = mean_absolute_error(y_pred, Y_test)
print(error)

plt.clf()
plt.scatter(X_test[:100], Y_test[:100], color='blue', label='Actual values')
plt.plot(range(len(y_pred[:100])), y_pred[:100], color='red', label='Predicted values', marker='o')
plt.legend()
plt.xlabel("Weight(kg)")
plt.ylabel("Height(cm)")
plt.savefig("./WW/")
# plt.show()