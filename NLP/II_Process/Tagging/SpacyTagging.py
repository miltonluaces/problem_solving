# pip install spacy && python -m spacy download en
# pip install spacy && python -m spacy download en_core_web_lg

from scipy import spatial
from nltk.chunk import conlltags2tree
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from spacy import displacy
from spacy.tokens import Token
from spacy.tokens import Doc


# Tokenizing
nlp = spacy.load('en')
doc = nlp('Hello     World!')
for token in doc:
    print('"' + token.text + '"')
 
# Word tagging
doc = nlp("Next week I'll   be in Madrid.")
for token in doc:
    print("\ntext    : " + token.text)
    print("idx     : " + str(token.idx))
    print("lemma   : " + token.lemma_)
    print("isPunct : " + str(token.is_punct))
    print("isSpace : " + str(token.is_space))
    print("shape   : " + str(token.shape_))
    print("pos     : " + str(token.pos_))
    print("tag     : " + token.tag_)

# Sentence tagging
doc = nlp("These are apples. These are oranges.")
for sent in doc.sents: print(sent)

# Part-of-speech tagging
doc = nlp("Next week I'll be in Madrid.")
print([(token.text, token.tag_) for token in doc])

# Named entity recognition tagging
doc = nlp("Next week I'll be in Madrid.") 
for ent in doc.ents: print(ent.text, ent.label_)

# In/Out/Begin (IOB) tagging
doc = nlp("Next week I'll be in Madrid.")
iob = [(token.text, token.tag_, "{0}-{1}".format(token.ent_iob_, token.ent_type_) if token.ent_iob_ != 'O' else token.ent_iob_) for token in doc]
print(iob)
print(conlltags2tree(iob)) # Tree

# Entity tagging
doc = nlp("I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ")
for ent in doc.ents: print(ent.text, ent.label_)
displacy.render(doc, style='ent', jupyter=True)

# Chunk tagging
doc = nlp("Wall Street Journal just published an interesting piece on crypto currencies")
for chunk in doc.noun_chunks: print(chunk.text, chunk.label_, chunk.root.text)

# Dependency parsing
for token in doc:
    print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

# Distance measuring
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})

# Word vectors
nlp = spacy.load('en_core_web_lg')
print(nlp.vocab['banana'].vector)

# Cosine similarity
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector
queen = nlp.vocab['queen'].vector
king = nlp.vocab['king'].vector
 
# Closest vector in the vocabulary (to the result of "man" - "woman" + "queen")
maybe_king = man - woman + queen
similarities = []
 
for word in nlp.vocab:
    if not word.has_vector: continue
    similarity = cosine_similarity(maybe_king, word.vector)
    similarities.append((word, similarity))

similarities = sorted(similarities, key=lambda item: -item[1])
print([w[0].text for w in similarities[:10]])

# Computing similarity
banana = nlp.vocab['banana']
dog = nlp.vocab['dog']
fruit = nlp.vocab['fruit']
animal = nlp.vocab['animal']

print(dog.similarity(animal), dog.similarity(fruit)) # 0.6618534 0.23552845
print(banana.similarity(fruit), banana.similarity(animal)) # 0.67148364 0.2427285

target = nlp("Cats are beautiful animals.")
 
doc1 = nlp("Dogs are awesome.")
doc2 = nlp("Some gorgeous creatures are felines.")
doc3 = nlp("Dolphins are swimming mammals.")
 
print(target.similarity(doc1))  # 0.8901765218466683
print(target.similarity(doc2))  # 0.9115828449161616
print(target.similarity(doc3))  # 0.7822956752876101

# Sentiment Analysis
sentAnalyzer = SentimentIntensityAnalyzer()
def polarity_scores(doc):
    return sentAnalyzer.polarity_scores(doc.text)
 
Doc.set_extension('polarity_scores', getter=polarity_scores)
 
nlp = spacy.load('en')
doc = nlp("Really Whaaat event apple nice! it!")
print(doc._.polarity_scores)
# {'neg': 0.0, 'neu': 0.596, 'pos': 0.404, 'compound': 0.5242}

nlp = spacy.load('en')
print(nlp.pipeline) 
 
def penn_to_wn(tag):
    if tag.startswith('N'): return 'n'
    elif tag.startswith('V'): return 'v' 
    elif tag.startswith('J'): return 'a'
    elif tag.startswith('R'): return 'r'
    return None
 
 
class WordnetPipeline(object):
    def __init__(self, nlp):
        Token.set_extension('synset', default=None)
 
    def __call__(self, doc):
        for token in doc:
            wn_tag = penn_to_wn(token.tag_)
            if wn_tag is None: continue
 
            ss = wn.synsets(token.text, wn_tag)[0]
            token._.set('synset', ss)
 
        return doc
 
 
nlp = spacy.load('en')
wn_pipeline = WordnetPipeline(nlp)
nlp.add_pipe(wn_pipeline, name='wn_synsets')
doc = nlp("Paris is the awesome capital of France.")
 
for token in doc:
    print(token.text, "-", token._.synset)

print(nlp.pipeline)
print(ent.text, ent.label_)


