#Init block
#%%
import os
os.chdir('D:/source/repos/ProblemSolving')
print(os.getcwd())

import recordlinkage
from recordlinkage.datasets import load_febrl1
#from Visual.Misc.Tables import ShowDataFrame

data = load_febrl1()
print(data.head(10))

# Indexation step
#%%

idxr = recordlinkage.Index()
idxr.block(left_on='given_name') # force matching in this field
candidateLinks = idxr.index(data)

# Comparison step
#%%

cmp = recordlinkage.Compare()
cmp.exact('given_name', 'given_name', label='given_name')
cmp.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
cmp.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
cmp.exact('suburb', 'suburb', label='suburb')
cmp.exact('state', 'state', label='state')
cmp.string('address_1', 'address_1', threshold=0.85, label='address_1')

features = cmp.compute(candidateLinks, data)

# Classification step
#%%

matches1 = features[features.sum(axis=1) >= 4]
matches2 = features[features.sum(axis=1) >= 5]
matches3 = features[features.sum(axis=1) >= 6]
print(matches1)
print(len(candidateLinks))
print(len(matches1))
print(len(matches2))
print(len(matches3))