import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Binance 거래소 API 사용
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", timeframe='1d', limit=1000)

# 가격 데이터를 DataFrame으로 변환
df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)

# ARIMA 모델 학습과 예측을 위해 가격 데이터의 차분(Differencing) 수행
df['close_diff'] = df['close'].diff().fillna(0)  # 차분 값 계산 및 결측치 처리

# ARIMA 모델 학습과 예측
model = ARIMA(df['close_diff'], order=(5, 1, 0))  # ARIMA(p, d, q) 파라미터 설정
results = model.fit()
forecast_steps = 300  # 예측할 기간 설정

# 마지막 학습 데이터에서부터 예측 기간만큼 미래를 예측
forecast_diff = results.forecast(steps=forecast_steps)
forecast_cumsum = forecast_diff.cumsum()  # 누적합을 통해 원래 가격으로 변환

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 학습 데이터의 가격 그래프 그리기
plt.plot(df.index, df['close'], label='Price', color='blue')

# ARIMA 예측 결과 그래프에 추가
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1)[1:]
plt.plot(forecast_index, forecast_cumsum + df['close'].iloc[-1], color='purple', label='Forecast')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Stock Price Forecast')
plt.legend()
plt.grid(True)
plt.show()
