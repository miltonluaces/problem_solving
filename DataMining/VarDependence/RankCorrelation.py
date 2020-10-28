from scipy.stats import spearmanr
from scipy.stats import kendalltau


a = [1,2,3,4,5]
b = [5,6,7,8,7]
rc = spearmanr(a,b)
print(rc)
print('coeff = ', '{0:.2f}'.format(rc[0]))
print('pVal = ', '{0:.2f}'.format(rc[1]))


# generate related variables
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
seed(1)
# prepare data
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
# plot
pyplot.scatter(data1, data2)
pyplot.show()

# Spearman's correlation
# Ordinal scale: r corr on ranks
coef, p = spearmanr(data1, data2)
print('\nSpearman correlation')
print('coeff = ', '{0:.2f}'.format(coef))
print('pVal = ', '{0:.2f}'.format(p))

	
# Kendall's correlation
# Normalized score of the correspondant ranks
coef, p = kendalltau(data1, data2)
print('\nKendall correlation')
print('coeff = ', '{0:.2f}'.format(coef))
print('pVal = ', '{0:.2f}'.format(p))
