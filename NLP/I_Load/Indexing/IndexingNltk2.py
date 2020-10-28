import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer 

# Inverted index datastructure 
class TextIndex:

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    # Lookup a word in the index
    def GetWord(self, idx):
        return self.documents.get(idx, None) 

    def GetIdx(self, word):
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)
        return self.index.get(word)
        
 
    def AddDoc(self, txtFile):
        txt = open(txtFile).read()
        tokens = word_tokenize(txt)
        for word in tokens:
            self.Add(word)

    def GetContext(self, idx, start, end):
            idxStart = idx-start
            idxEnd = idx+end+1
            sent = []
            for i in range(idxStart,idxEnd):
                sent.append(self.GetWord(i))
            return sent

    def Add(self, document):
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue
 
            if self.stemmer:
                token = self.stemmer.stem(token)
 
            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)
 
        self.documents[self.__unique_id] = document
        self.__unique_id += 1           
 
 

ti = TextIndex(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))
  
fileName = 'D:/data/text/wonderland.txt'
ti.AddDoc(fileName)

print('Index: ' , ti.GetIdx('inches'))
print('Word: ', ti.GetWord(1566))

print('Context: ' , ti.GetContext(1566, 3, 4))