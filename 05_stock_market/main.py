import numpy as np
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = read_csv('05_stock_market/Stock Market Dataset.csv')
print(df.info())

y = df['Unnamed: 0']
x = df.loc[:, (df.columns != 'Unnamed: 0') & (df.columns != 'Date')]

for i in x:
    for (idx, j) in enumerate(x[i]):
        x[i][idx] = str(j).replace(',','')
    x[i].astype(np.float64)
# plt.plot(x, y)
# plt.show()

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

j=0
for (idx, i) in enumerate(x):
    if j==10:
        break
    else: j+=1
    plt.plot(y, x[idx])
plt.show()