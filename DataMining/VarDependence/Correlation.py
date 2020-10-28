from scipy.stats.stats import pearsonr   

a = [1,4,6]
b = [1,2,3]   

pearson = pearsonr(a,b)
print('r = ', '{0:.2f}'.format(pearson[0]))
print('r2 = ', '{0:.2f}'.format(pearson[0]**2))
print('pVal = ', '{0:.2f}'.format(pearson[1]))


# Adjusted R squared
# Bonferroni penalization (Occams razor)