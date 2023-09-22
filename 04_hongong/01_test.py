import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

arr = [1, 50, 60, 4, 5, 55]
ar = [10, 510, 620, 40, 50, 530]

plt.scatter(arr, ar)
# plt.show()

target = [1, 0, 0, 0, 1, 1]
new = [[l, w] for l, w in zip(arr, ar)]

kn = KNeighborsClassifier(n_neighbors=2)
kn.fit(new, target)
print(kn.score(new, target))
print(kn.predict([[60, 500]]))

from sklearn.model_selection import train_test_split
