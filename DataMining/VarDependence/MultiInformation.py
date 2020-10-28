import dit as dt
import dit.multivariate as dm
from dit.multivariate import caekl_mutual_information as Caekl

# Total correlation
d = dt.example_dists.Xor()
tc1 = dm.total_correlation(d)
tc2 = dm.total_correlation(d, rvs=[[0], [1]])

print(tc1)
print(tc2)

# Dual Total correlation (binding information)
d = dt.Distribution(['000', '111'], [1/2, 1/2])
tc = dm.total_correlation(d)
bc = dm.binding_information(d)

from dit.multivariate import caekl_mutual_information as J

# Caekl Mutual information
c = Caekl(d)
print(c)
