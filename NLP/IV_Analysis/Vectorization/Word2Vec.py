# WORD EMBEDDINGS WITH WORD2VEC

# Imports

from nltk import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
import warnings 
warnings.filterwarnings(action = 'ignore') 

# Load data
sample = open("D:\\Data\\text\\wonderland.txt", "r") 
s = sample.read() 
f = s.replace("\n", " ") 

# Tokenize
 
data = [] 
for i in sent_tokenize(f): 
	sent = [] 
	for j in word_tokenize(i): 
		sent.append(j.lower()) 
	data.append(sent) 

# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5) 

# Results
print("CBOW results")
print("Similarity(alice, wonderland) : ",  model1.similarity('alice', 'wonderland')) 
print("Similarity(alice, machines): ", model1.similarity('alice', 'machines')) 

# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5, sg=1) 

# Print results 
print("Similarity(alice, wonderland) : ",  model2.similarity('alice', 'wonderland')) 
print("Similarity(alice, machines): ", model2.similarity('alice', 'machines')) 
