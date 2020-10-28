from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize, regexp_tokenize, wordpunct_tokenize, blankline_tokenize
# All text in one
str = "Want to see how this example works. I'm not sure it will work at all. Do you?"
tokens = sent_tokenize(str)
print(tokens)

# Simple splitting
tokens = str.split()
print(tokens)

# Word splitting (a bit smarter)
tokens = word_tokenize(str)
print(tokens)

# Regex tokenizing
tokens = regexp_tokenize(str, pattern='\w+')
print(tokens)

# wordpunct tokenize (punctuation separated)
tokens = wordpunct_tokenize(str)
print(tokens)