import numpy
import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Feature Extraction with PCA

# Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# Feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)

# Results
print("Explained Variance:", fit.explained_variance_ratio_)
print(fit.components_)


#df = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')
#df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#df.dropna(how="all", inplace=True) # drops the empty line at file-end
#df.tail()
#X = df.ix[:,0:4].values
#y = df.ix[:,4].values
#X_std = StandardScaler().fit_transform(X)

## Traces
#traces = []
#for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
#    marker = Marker(size=12, line=Line(color='rgba(217, 217, 217, 0.14)', width=0.5),opacity=0.8)
#    trace = Scatter(x=Y[y==name,0], y=Y[y==name,1], mode='markers', name=name, marker=marker)
#    traces.append(trace)

## Plot
#data = Data(traces)
#layout = Layout(xaxis=XAxis(title='PC1', showline=False), yaxis=YAxis(title='PC2', showline=False))
#fig = Figure(data=data, layout=layout)
#py.iplot(fig)
