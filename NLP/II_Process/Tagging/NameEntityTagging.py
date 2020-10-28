import nltk 
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk import pos_tag
from nltk import word_tokenize
from nltk import ne_chunk

str = "Milton is working in Accenture in Dublin"

tags = ne_chunk(pos_tag(word_tokenize(str)), binary=False)
print(tags)

# NERTagger from Stanford
