import re
from collections import Counter


# All words in lower case
def GetWords(text): 
    return re.findall(r'\w+', text.lower())

# Data
words = Counter(GetWords(open('Data/big.txt').read()))

# Probability of w
def P(w, n=sum(words.values())): 
    return words[w] / n

# Most probable spelling correction for word
def correction(w): 
    return max(candidates(w), key=P)

# Generate possible spelling corrections for word
def candidates(w): 
    return (known([w]) or known(edits1(w)) or known(edits2(w)) or [w])

# The subset of ws that appear in the dictionary of words
def known(w): 
    return set(w for w in words if w in words)

# All edits that are one edit away from w
def edits1(w):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

# All edits that are two edits away from w
def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
