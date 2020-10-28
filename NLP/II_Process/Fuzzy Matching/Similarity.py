import os
import csv
import nltk
import json
import math
import random
import distance 

# SIMILARITY SCORES

# Important Paths
fixtures = os.path.join(os.getcwd(), "fixtures")
products = os.path.join(fixtures, "products")

# Module Constants
googId   = 'http://www.google.com/base/feeds/snippets'

# Create a generator to load data from the products data source.
def LoadData(name):
    with open(os.path.join(products, name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: yield row

def GoogleKey(key):
    return os.path.join(googId, key)

# Load Datasets into Memory
amazon  = list(LoadData('amazon.csv'))
google  = list(LoadData('google.csv'))
mapping = list(LoadData('perfect_mapping.csv'))

# Datasets contents
for name, dataset in (('Amazon', amazon), ('Google Shopping', google)):
    print("{} dataset contains {} records".format(name, len(dataset)))
    print( "Record keys: {}\n".format(", ".join(dataset[0].keys())))

# Datasets contents matching
print( "There are {} matching records to link".format(len(mapping)))

# Convert dataset to records indexed by their Id and show dumps (examples)
amazon  = dict((v['id'], v) for v in amazon)
google  = dict((v['id'], v) for v in google)
X = amazon['b0000c7fpt']
Y = google[GoogleKey('17175991674191849246')]
print(json.dumps(X, indent=2))
print(json.dumps(Y, indent=2))

# Typographic Distances
print(distance.levenshtein("lenvestein", "levenshtein"))
print(distance.hamming("hamming", "hamning"))

# Example: Compare glyphs, syllables, or phonemes 
t1 = ("de", "ci", "si", "ve")
t2 = ("de", "ri", "si", "ve")
print(distance.levenshtein(t1, t2))

# Sentence Comparison
sent1 = "The quick brown fox jumped over the lazy dogs."
sent2 = "The lazy foxes are jumping over the crazy Dog."
print(distance.nlevenshtein(sent1.split(), sent2.split(), method=1))

# Normalization
print(distance.hamming("fat", "cat", normalized=True))
print(distance.nlevenshtein("abc", "acd", method=1))  # shortest alignment
print(distance.nlevenshtein("abc", "acd", method=2))  # longest alignment

# Set measures
print(distance.sorensen("decide", "resize"))
print(distance.jaccard("decide", "resize"))


# PROCESSED TEXT SCORE

# Returns a similarity vector of match scores: [name_score, description_score, manufacturer_score, price_score]
def Tokenize(sent):
    lemmatizer = nltk.WordNetLemmatizer() 
    for token in nltk.wordpunct_tokenize(sent):
        token = token.lower()
        yield lemmatizer.lemmatize(token)

def NormalizedJaccard(*args):
    try: return distance.jaccard(*[Tokenize(arg) for arg in args])
    except UnicodeDecodeError: return 0.0

print(NormalizedJaccard(sent1, sent2))


# SIMILARITY VECTORS

# Returns a similarity vector of match scores: [name_score, description_score, manufacturer_score, price_score]
def Similarity(prod1, prod2):
    pair  = (prod1, prod2)
    names = [r.get('name', None) or r.get('title', None) for r in pair]
    descr = [r.get('description') for r in pair]
    manuf = [r.get('manufacturer') for r in pair]
    price = [float(r.get('price')) for r in pair]
    return [NormalizedJaccard(*names), NormalizedJaccard(*descr), NormalizedJaccard(*manuf), abs(1.0/(1+ (price[0] - price[1]))),]

print(Similarity(X, Y))

# WEIGHTED PAIRWISE MATCHING

thres = 0.90
weights   = (0.6, 0.1, 0.2, 0.1)

matches = 0
for azprod in amazon.values():
    for googprod in google.values():
        vector = Similarity(azprod, googprod)
        score  = sum(map(lambda v: v[0]*v[1], zip(weights, vector)))
        if score > thres:
            matches += 1
            print("{0:0.3f}: {1} {2}".format(score, azprod['id'], googprod['id'].split("/")[-1]))

print("\n{} matches discovered".format(matches))



