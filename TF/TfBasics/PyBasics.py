import numpy
import scipy

# Loop and identation
for i in range(1,11):
    print(i)
    if i == 5:
        break
    
# Lists
a = [1, 2.2, 'python']
print(a)

# Dictionary
d = {1:'value','key':2}
print("d[1] = ", d[1]);
print("d['key'] = ", d['key']);

# Time series
ts = [6, 8, 12, 35, 24, 15, 7, 33, 46, 28, 62, 17, 54, 42]
print (ts)
print(ts[4])


# SCIPY 
# Derivatives and integrals
p = poly1d([3,4,5]); print(p)
d = p.deriv(); print(d)
i = d.integ(k=6); print(i)

# PANDAS
names = ['a','b','c','d','e']
values = [10, 20, 30, 40, 50]
ranges = list(zip(names, values))
Ranges = pd.DataFrame(data=ranges, columns =[ 'Names', 'Values'])
Ranges.to_csv('ranges.csv', index=False , header=False)

ranges = {10:'a', 20:'b', 30:'c', 40:'d', 50:'e', 100:'f'}

for key,val in ranges.items():
    print(key, '->', val)

tsc = TSCat(ranges)

# NLTK
import nltk
#nltk.download()
from nltk.book import *
print(" ")
#text1
#text1.concordance("monstrous")
#text1.similar("monstrous")
#text2.common_contexts(["monstrous", "very"])
#print(len(text3))
print(len(set(text3)) / len(text3))

