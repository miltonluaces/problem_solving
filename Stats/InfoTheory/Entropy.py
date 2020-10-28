import dit
from dit.divergences import cross_entropy

d = dit.Distribution(['H'], [1])
e = dit.shannon.entropy(d)
print(e)

# So since we know that the outcome from our distribution will always be H, we have to ask zero questions to figure that out. If however we have a fair coin:

d = dit.Distribution(['H', 'T'], [1/2, 1/2])
e = dit.shannon.entropy(d)
print(e)

#The entropy tells us that we must ask one question to determine whether an H or T was the outcome of the coin flip. Now what if there are three outcomes? Let’s consider the following situation:

d = dit.Distribution(['A', 'B', 'C'], [1/2, 1/4, 1/4])
e = dit.shannon.entropy(d)
print(e)


# Cross entropy
p = dit.Distribution(['0', '1'], [1/2, 1/2])

ce = cross_entropy(p, p)