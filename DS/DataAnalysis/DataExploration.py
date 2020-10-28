import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from Utils.Admin.Standard import *

df = pd.read_csv(csvPath + "loanTrain.csv")

print(df.head(10))

print(df.columns)
print(df.shape)

print(df.describe())
df['ApplicantIncome'].describe() 

df['Loan_Status'].value_counts(normalize = 'True')
df['Loan_Status'].value_counts().plot.bar(title = 'Loan_Status')
plt.show()

sns.distplot(df['ApplicantIncome'])
plt.show()

df['ApplicantIncome'].plot.box()
plt.show()