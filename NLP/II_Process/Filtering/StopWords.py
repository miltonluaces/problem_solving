from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stop = set(stopwords.words('english'))
print("Stopwords")
print(stop)
print()

words = word_tokenize(data)

wordsFilt = []
for w in words:
    if w not in stop: wordsFilt.append(w)
 
print(wordsFilt)