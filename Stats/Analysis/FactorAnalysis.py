from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis

X, _ = load_digits(return_X_y=True)
transformer = FactorAnalysis(n_components=7, random_state=0)
Xtr = transformer.fit_transform(X)
print(X.shape)
print(Xtr.shape)
