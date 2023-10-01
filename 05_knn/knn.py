import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def dividedArrToKNNArr(xArr, yArr):
    result = []
    for i in range(len(xArr)):
        result.append(KNNPos(xArr[i], yArr[i]))
    
    return result

class KNNPos:
    def __init__(self, x, y, d = 0):
        self.x = x
        self.y = y
        self.d = d

    def getPos(self):
        return (self.x, self.y)

    def getKNNPos(self):
        return (self.x, self.y, self.d)

class KNNArr:
    def __init__(self, arr):
        self.arr: [KNNPos] = arr

    def getArr(self):
        return self.arr

    def getXArr(self):
        return [i.x for i in self.arr]
    
    def getYArr(self):
        return [i.y for i in self.arr]
    
    def getDArr(self):
        return [i.d for i in self.arr]
    
    def getDividedArr(self):
        return (self.getXArr(), self.getYArr(), self.getDArr())

class KNNPredictor:
    def __init__(self):
        self.trainArr = []
        self.testArr = []
        

    def registerTrainSet(self, arr):
        knnArr = dividedArrToKNNArr(arr)
        self.trainArr.append(knnArr)
    
    def registerTestSet(self, arr):
        knnArr = dividedArrToKNNArr(arr)
        self.testArr.append(knnArr)

    def train(self):
        for i in range(len(self.trainArr)):
            pass
            # self.trainArr[i].d = 


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
