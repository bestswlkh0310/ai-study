from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
import matplotlib.pyplot as plt
import random
# print(diabetes.data.shape, diabetes.target.shape)

# plt.scatter(diabetes.data[:, 2], diabetes.target)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

x = diabetes.data[:, 2]
y = diabetes.target

w = random.uniform(0, 1) # random
b = random.uniform(0, 1) # random

###########

y_hat = x[0] * w + b # 첫 번째 샘플에 대한 예측값
print("y-hat", y_hat)
print("y-real", y[0])
print('loss', y_hat-y[0]) # 상상과 현실..의 갭

w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
print("y-hat-inc", y_hat_inc)
print('loss-inc', y_hat_inc-y[0]) # 쪼오오오금 줄어들었어요

############

w_rate = (y_hat_inc - y_hat) \
       / (w_inc - w) # 증가량의 변화율
       # x[0]과 같다
print('w-rate', w_rate)

print('w', w)
w_new = w + w_rate
print('w-new',w_new)

b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
print('y-hat-inc', y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print('b-rate',b_rate)

b_new = b + 1
print('b-new',b_new)

err = y[0] - y_hat # loss
w_new = w + w_rate * err # ADD: * err
b_new = b + 1 * err # ADD: * err
print('w-new, b-new', w_new, b_new)

y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print('w-new, b-new', w_new, b_new)

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print('w, b',w, b)

# plt.scatter(x, y)
# pt1 = (-0.1, -0.1 * w + b)
# pt2 = (0.15, 0.15 * w + b)
# plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

for i in range(1, 1000):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print(w, b)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

plt.scatter(x, y)
for i in range(100):
    x_new = random.uniform(-0.1, 0.1)
    y_pred = x_new * w + b
    plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

