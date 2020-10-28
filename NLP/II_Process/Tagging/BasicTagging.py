from nltk import pos_tag
from nltk import word_tokenize

str = "The glass is on the table"
words = word_tokenize(str)
tags = pos_tag(words)
print(tags)

# List all nouns
nouns = []
for word, pos in tags:
    if pos in ['NN', 'NNP']: nouns.append(word)
 
print(nouns)
