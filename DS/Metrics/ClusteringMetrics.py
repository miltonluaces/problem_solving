from sklearn.metrics.cluster import homogeneity_score

hs = homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0]); print(hs)
