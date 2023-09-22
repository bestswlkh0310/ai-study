import ccxt 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", timeframe='1d', limit=100)

df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
# print(df.index)
# plt.plot(df.index, df['open'])

# 그래프 그리기
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# 봉 차트 그리기
for i in range(len(df)):
    date = df.index[i]
    open_price = df['open'][i]
    high_price = df['high'][i]
    low_price = df['low'][i]
    close_price = df['close'][i]
    
    color = 'b' if close_price < open_price else 'r'  # 종가가 개장가보다 작으면 빨간색, 크면 초록색
    plt.plot([date, date], [low_price, high_price], color=color)  # 최저가부터 최고가까지 세로선
    plt.plot([date, date], [open_price, close_price], color=color, linewidth=5)  # 개장가부터 종가까지 세로선

plt.show()