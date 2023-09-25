from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
import matplotlib.pyplot as plt
import random

x = diabetes.data[:, 2]
y = diabetes.target

w = random.uniform(0, 1) # random
b = random.uniform(0, 1) # random

# training

for i in range(1, 1000):
    for x_i, y_i in zip(x, y):
        y_hat = x_i ** 2 + x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w += w_rate * err
        b += 1 * err
print(w, b)

# show

import numpy as np
plt.scatter(x, y)
a = np.arange(-0.1, 0.2, 0.1)
y_pred = a * w + b
plt.plot(a, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
