from Utils.Admin.Standard import *
from Parser import Parse
import Parser as pr
import WCalculator as wc
import TxtGenerator as tg


# Load text and lowercase it
txtFile = txtPath + 'wonderlandSample.txt'
txt = open(txtFile).read()
txt = txt.lower()

# Parser
convDict, nChars, nAlpha = pr.Parse(txt)
char2int = convDict['char2int']
print("Total Characters: ", nChars)
print("Total Alphabet: ", nAlpha)
print("Char To Int Dictionary: ", char2int)

# Preprocess
seqSize = 400
X, y, dataX = pr.Preprocess(txt, seqSize, nChars, nAlpha, char2int)
print("X is ", X)
print( "y is ", y)

# Weight calculation
weightFile = modelPath + 'lstmWeights.hdf5'
open(weightFile, encoding="utf8", errors='ignore').read()
wc.CalcWeights(txt)

# Text generation
#tg.Generate(txt, weights=None)

