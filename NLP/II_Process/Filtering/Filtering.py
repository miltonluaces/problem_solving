import nltk
wordsFilt = []
for w in words:
    if w not in rare: wordsFilt.append(w)
 
print(wordsFilt)
t

dist = FreqDist(token)
rare = dist.keys()[-50:]

