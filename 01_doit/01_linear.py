from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
import matplotlib.pyplot as plt
import random
import numpy as np

# 1. 데이터 불러오기
x = diabetes.data[:, 2]
y = diabetes.target

w = random.uniform(0, 1)
b = random.uniform(0, 1)

# 2. 데이터 학습
for i in range(1000): # 1000번 에포크
    for x_i, y_i in zip(x, y): # 학습
        y_hat = x_i * w + b # 예측값
        err = y_i - y_hat # 관측값 - 예측값
        w_rate = x_i
        w = w + w_rate * err
        print(w, w_rate) if i == 0 else None
        b = b + 1 * err
print(w, b)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')

plt.scatter(x, y)
for i in range(100):
    x_new = random.uniform(-0.1, 0.15)
    y_pred = x_new * w + b
    plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

