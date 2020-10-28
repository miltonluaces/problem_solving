import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk.corpus  
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text

txtFile = 'D:/data/text/wonderlandSample.txt'
txt = open(txtFile).read()
tokens = word_tokenize(txt)
textList = Text(tokens)
conc = textList.concordance('Alice'); print(conc)
conc = textList.concordance('inches'); print(conc)
conc = textList.concordance('golden'); print(conc)

