import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

# Binance 거래소 API 사용
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("ETH/USDT", timeframe='1d', limit=1000)

# 가격 데이터를 DataFrame으로 변환
df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
# 데이터 스케일링
scaler = MinMaxScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])

# LSTM 학습에 사용할 sequence 데이터 준비
sequence_length = 800  # Set sequence length to 800

sequences = []
for i in range(len(df) - sequence_length):
    sequence = df['close_scaled'].values[i:i+sequence_length]
    sequences.append(sequence)

X = []
y = []
for seq in sequences:
    X.append(seq[:-1])  # 마지막 데이터를 제외한 데이터를 입력으로 사용
    y.append(seq[-1])   # 마지막 데이터를 타겟으로 사용

X = np.array(X)
y = np.array(y)

# LSTM 모델 생성
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(X.shape[1], 1), return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='linear'))

# Use Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# LSTM 모델 학습
model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])

# 장기적인 예측을 위해 마지막 시퀀스 데이터로부터 미래를 예측
forecast_steps = 300
last_sequence = df['close_scaled'].values[-sequence_length:]
forecast = []

for _ in range(forecast_steps):
    input_data = last_sequence[-sequence_length + 1:]
    input_data = input_data.reshape(1, sequence_length - 1, 1)

    prediction = model.predict(input_data)[0, 0]
    forecast.append(prediction)
    last_sequence = np.append(last_sequence, prediction)[-sequence_length:]

forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# 미래 예측 결과를 DataFrame에 추가
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1)[1:]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['close'])

# 원래 가격 데이터와 예측 결과를 하나로 합치기
merged_df = pd.concat([df[['close']], forecast_df])

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 학습 데이터의 가격 그래프 그리기
plt.plot(merged_df.index[:-forecast_steps], merged_df['close'][:-forecast_steps], label='Price', color='blue')

# LSTM 예측 결과 그래프에 추가
plt.plot(merged_df.index[-forecast_steps:], merged_df['close'][-forecast_steps:], color='purple', label='Forecast')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('LSTM Stock Price Forecast')
plt.legend()
plt.grid(True)
plt.show()
