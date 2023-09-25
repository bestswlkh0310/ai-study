import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def scatting(n, x1, x2, y1, y2):
    x_arr1 = []
    for _ in range(n):
        x1 = random.random() * random.uniform(min(x1, x2), max(x1, x2))
        x_arr1.append(x1)

    y_arr1 = []
    for _ in range(n):
        y1 = random.random() * random.uniform(min(x1, x2), max(x1, x2))
        y_arr1.append(y1)
    return x_arr1, y_arr1
x_arr1, y_arr1 = scatting(100, 1, 3, 1, 3)
x_arr2, y_arr2 = scatting(100, 0, 1, 0, 1)

test_pos = (0.4, 1.5)

def distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def getNearestNeighbor(pos, arrs, k):
    ar = []
    for el in arrs:
        xArr, yArr = el
        for p in range(len(xArr)):
            d = distance(pos[0], xArr[p], pos[1], yArr[p])
            ar.append(((xArr[p], yArr[p]), d))
    ar = sorted(ar, key=lambda x: x[1])
    return ar[:k]
    
v = getNearestNeighbor(test_pos, [
    (x_arr1, y_arr1),
    (x_arr2, y_arr2)
    ], 10)

x1 = []
y1 = []
mx = 0

for i in v:
    x1.append(i[0][0])
    y1.append(i[0][1])
    if i[1] > mx:
        mx = i[1]

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(x_arr1, y_arr1)
ax.scatter(x_arr2, y_arr2)

ax.scatter(test_pos[0], test_pos[1])

ax.scatter(x1, y1)
ax.add_patch(patches.Circle(test_pos, mx, alpha=0.3))
plt.show()
