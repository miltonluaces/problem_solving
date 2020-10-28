from outliers import smirnov_grubbs as grubbs
import pandas as pd

data = pd.Series([1, 8, 9, 10, 9])
gt = grubbs.test(data, alpha=0.05)
print(gt)

#Jacknife
