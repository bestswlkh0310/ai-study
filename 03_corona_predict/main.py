from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from util import create_data_set

# 1. 파일 불러오기
df = read_csv('03_corona_predict/corona_daily.csv', usecols=[3], engine='python', skipfooter=3)
df.info()

# 2. 0~1 사이의 값으로 정규화
data_set = df.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
Data_set = scaler.fit_transform(data_set)

# 3. 데이터셋 만들기
train_data, test_data = train_test_split(Data_set, test_size=0.2, shuffle=False)

look_back = 3
x_train, y_train = create_data_set(train_data, look_back)
x_test, y_test = create_data_set(test_data, look_back)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print(X_train.shape, X_test.shape)

# 4. 모델 설계
model = Sequential(name='coronada')
model.add(SimpleRNN(3, input_shape=(1, look_back)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()

# 5. 학습
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

# 6. 예측
# inverse_transform - 정규화 to 원래 데이터 형식
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict2 = scaler.inverse_transform(train_predict)
test_predict2 = scaler.inverse_transform(test_predict)
Y_train = scaler.inverse_transform([y_train])
Y_test = scaler.inverse_transform([y_test])

# 평균 제곱근 오차
train_score = math.sqrt(mean_squared_error(Y_train[0], train_predict2[:,0]))
print(train_score)

test_score = math.sqrt(mean_squared_error(Y_test[0], test_predict2[:,0]))
print(test_score)

train_predict_plot = np.empty_like(data_set)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back: len(train_predict2) + look_back, :] = train_predict2

test_predict_plot = np.empty_like(data_set)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict2) + look_back * 2 : len(data_set), :] = test_predict2

plt.plot(data_set)
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()