from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize

str = "A passenger plane has crashed shortly after take-off from Kyrgyzstan's capital, Bishkek, killing a large number of those on board. The head of Kyrgyzstan's civil aviation authority said that out of about 90 passengers and crew, only about 20 people have survived. The Itek Air Boeing 737 took off bound for Mashhad, in north-eastern Iran, but turned round some 10 minutes later."

tagger = StanfordPOSTagger('FileModels/Postagger/models/english-bidirectional-distsim.tagger', 'FileModels/Postagger/stanford-postagger.jar')
tokens = word_tokenize(str)
tags = tagger.tag(tokens)
print(tags)


