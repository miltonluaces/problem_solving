import chart_studio.plotly as py
import plotly.figure_factory as ff
import numpy as np

X = np.random.rand(15, 15)
dendro = ff.create_dendrogram(X)
dendro['layout'].update({'width':800, 'height':500})
py.iplot(dendro, filename='simple_dendrogram')
