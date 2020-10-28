from Utils.Admin.Standard import *
import random
import pickle
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
import spacy
spacy.load('en')
from spacy.lang.en import English
import gensim
from gensim import corpora 
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim


parser = English()
def Tokenize(text):
    ldaTokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace(): continue
        elif token.like_url: ldaTokens.append('URL')
        elif token.orth_.startswith('@'): ldaTokens.append('SCREEN_NAME')
        else: ldaTokens.append(token.lower_)
    return ldaTokens

def GetLemma(word):
    lemma = wn.morphy(word)
    if lemma is None: return word
    else: return lemma
    
def PreProcess(text):
    tokens = Tokenize(text)
    tokens = [t for t in tokens if len(t) > 4]
    tokens = [t for t in tokens if t not in en_stop]
    tokens = [GetLemma(t) for t in tokens]
    return tokens


data = []
with open(txtPath + 'dataset.txt') as f:
    for line in f:
        tokens = PreProcess(line)
        if(random.random() > 0.99): print(tokens); data.append(tokens)


dict = Dictionary(data)
corpus = [dict.doc2bow(text) for text in data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dict.save('dict.gensim')

numTopics = 5
ldamodel = LdaModel(corpus, num_topics = numTopics, id2word=dict, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
print('\nTopics:')
for topic in topics: print(topic)

newDoc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
newDoc = PreProcess(newDoc)
newDocBow= dict.doc2bow(newDoc)
print(newDocBow)
topics = ldamodel.get_document_topics(newDocBow)
print('\nTopics:')
print(topics)

dict = Dictionary.load('dict.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = LdaModel.load('model5.gensim')
ldaDisplay = pyLDAvis.gensim.prepare(lda, corpus, dict, sort_topics=False)
pyLDAvis.show(ldaDisplay)




#def GetLemma2(word):
    #return WordNetLemmatizer().lemmatize(word)
