from sklearn.cross_decomposition import CCA
#from misc import rcca
import numpy as np

U = np.random.random_sample(500).reshape(100,5)
V = np.random.random_sample(500).reshape(100,5)

cca = CCA(n_components=1)
U_c, V_c = cca.fit_transform(U, V)

res = np.corrcoef(U_c.T, V_c.T)[0,1]
print(res)

print(U_c.shape)                         # (100,1)
print(V_c.shape)                         # (100,1)
