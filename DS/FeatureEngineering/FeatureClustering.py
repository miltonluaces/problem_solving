import numpy as np
from sklearn import datasets, cluster

# Load data
digits = datasets.load_digits()
images = digits.images
#print(images)
print(images.shape)
X = np.reshape(images, (len(images), -1))
print(X.shape)

agglo = cluster.FeatureAgglomeration(n_clusters=32)
agglo.fit(X) 
cluster.FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward', memory=None, n_clusters=32, pooling_func=...)
XReduced = agglo.transform(X)
print(XReduced.shape)

# Johnson-Lindenstrauss lemma
rn = np.random.standard_normal(11000*45)
rn = rn.reshape(11000,45)
trMatrix = np.asmatrix(rn)


