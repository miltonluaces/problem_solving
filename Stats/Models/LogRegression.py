from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
pred1 = clf.predict(X[:2, :]) ; print(pred1)
pred2 = clf.predict_proba(X[:2, :]) ; print(pred2)
score =clf.score(X, y); print(score)