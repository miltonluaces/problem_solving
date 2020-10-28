# Suppose you have 2 data sets from unknown distribution and you want to test if some arbitrary statistic (e.g 7th percentile) is the same in the 2 data sets
# We could test the difference between the 7th percentile, and if we knew the null distribution of this statisic, we could test for the null hypothesis that the statistic = 0
# Permuting the labels of the 2 data sets allows us to create the empirical null distribution

import numpy as np
import matplotlib.pyplot as plt

# Distributions x, y
x = np.concatenate([np.random.exponential(size=200), np.random.normal(0, 1, size=100)])
y = np.concatenate([np.random.exponential(size=250), np.random.normal(0, 1, size=50)])

# Procedure
n1, n2 = map(len, (x, y))
reps = 10000
data = np.concatenate([x, y])
ps = np.array([np.random.permutation(n1+n2) for i in range(reps)])
xp = data[ps[:, :n1]]
yp = data[ps[:, n1:]]

samples = np.percentile(xp, 7, axis=1) - np.percentile(yp, 7, axis=1)
test_stat = np.percentile(x, 7) - np.percentile(y, 7)

# Plot
plt.hist(samples, 25, histtype='step')
plt.axvline(test_stat)
plt.axvline(np.percentile(samples, 2.5), linestyle='--')
plt.axvline(np.percentile(samples, 97.5), linestyle='--')
plt.show()
pVal = 2*np.sum(samples >= np.abs(test_stat))/reps
print("p-value =", pVal)
