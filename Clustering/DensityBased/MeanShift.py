from sklearn.cluster import MeanShift
import numpy as np

X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
clustering = MeanShift(bandwidth=2).fit(X)
print(clustering.labels_)

pred = clustering.predict([[0, 0], [5, 5]])
print(pred)
print(clustering)

