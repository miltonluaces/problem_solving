import pandas as pd
import recordlinkage as rl

# Data
names1 = ['alfred', 'bob', 'calvin', 'hobbes', 'rusty']
names2 = ['alfred', 'danny', 'callum', 'hobie', 'rusty']

# Convert to DataFrames
df1 = pd.DataFrame(pd.Series(names1, name='names'))
df2 = pd.DataFrame(pd.Series(names2, name='names'))

# Full index
indexer = rl.FullIndex()
candidateLinks = indexer.index(df1, df2)
print(candidateLinks)

# Blocked index
indexer = rl.BlockIndex(on='names')
candidateLinks = indexer.index(df1, df2)
print(candidateLinks)

# Random index
indexer = rl.RandomIndex(n=1)
candidateLinks = indexer.index(df1, df2)
print(candidateLinks)

# Sorted Neighborhood index
indexer = rl.SortedNeighbourhoodIndex(on='names', window=3)
candidateLinks = indexer.index(df1, df2)
print(candidateLinks)

# WHATS NEXT

# Compare
candidateLinks = indexer.index(df1, df2)
comp = rl.Compare(candidateLinks, df1, df2)
print(comp)


# Generate dataset with 1000 samples, 100 features and 2 classes

def gen_dataset(n_samples=1000, n_features=100, n_classes=2, random_state=123):  
    X, y = datasets.make_classification(
        n_features=n_features,
        n_samples=n_samples,  
        n_informative=int(0.6 * n_features),    # the number of informative features
        n_redundant=int(0.1 * n_features),      # the number of redundant features
        n_classes=n_classes, 
        random_state=random_state)
    return (X, y)

X, y = gen_dataset(n_samples=1000, n_features=100, n_classes=2)

# Train / test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)