import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
df = pd.DataFrame({'normal': np.random.normal(10, 3, 1000), 'chi': np.random.chisquare(4, 1000)})
print(df.head(50))

nBins = 8
pd.cut(df['normal'], nBins)
pd.cut(df['chi'], nBins)

bins = np.linspace(0, 20, nBins+1)
print('Linear ', bins)

df['normal'] = pd.cut(df['normal'], bins)
df['chi'] = pd.cut(df['chi'], bins, labels=['Level1','Level2','Level3','Level4','Level5','Level6','Level7','Level8'])
print(df.head())

plt.style.use('ggplot')

sizeNormal = df.groupby('normal').size()
sizeChi = df.groupby('chi').size()

cats = df['normal'].cat.categories
idx = np.array([x for x, _ in enumerate(cats)])
width = 0.35       
plt.bar(idx, sizeNormal, width, label='Normal')
plt.bar(idx + width, sizeChi, width, label='Chi Square')

plt.xticks(idx + width / 2, cats)
plt.legend(loc='best')
plt.xticks(rotation = 90)
plt.show()

# Logarithmic binning
df = pd.DataFrame({'normal': np.random.normal(10, 3, 1000), 'chi': np.random.chisquare(4, 1000)})

# Specifying the number of bins
bins2 = np.logspace(np.log10(np.min(df['normal'])), np.log10(np.max(df['normal'])), num=8)
print("\nLogarithmic")
print(bins2)

# Specifying the multiplier
multiplier = 2.5
bins3 = [np.min(df['normal'])]
curValue = bins3[0]
while curValue < np.max(df['normal']):
    curValue = curValue * multiplier
    bins3.append(curValue)

plt.plot(bins)
plt.plot(bins2)
plt.plot(bins3)
plt.show()

# Binning by equal number in each bin
def EqualNumBinning(x, nBins):
    n = len(x)
    return np.interp(np.linspace(0, n, nBins + 1), np.arange(n), np.sort(x))

x = np.random.randn(100)
n, bins, patches = plt.hist(x, EqualNumBinning(x, 10))
print("n")
print(n)
print("bins")
print(bins)
print("patches")
print(patches)

from sklearn import KBinsDiscretizer

X = np.array([[ -3., 5., 15 ], [  0., 6., 14 ], [  6., 3., 11 ]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
est.transform(X)  