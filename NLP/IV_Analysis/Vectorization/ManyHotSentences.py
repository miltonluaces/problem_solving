import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
sent1 = ['cold', 'cold', 'warm', 'cold']
sent2 = ['hot', 'hot', 'warm', 'cold', 'warm', 'hot']
sent3 = ['cold', 'warm','warm', 'hot']
sent4 = ['warm', 'hot']
doc = [sent1, sent2, sent3, sent4]
print('\nSentences')
print(doc)
words = list(np.concatenate(doc))
print('\nWord vector')
print(words)
words = list(set(words))
n = len(words)
dictOfWords = { words[i] : i for i in range(0, len(words) ) }
print(dictOfWords)

for sent in doc:


