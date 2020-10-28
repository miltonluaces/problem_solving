import numpy
from tensorflow.keras.utils import to_categorical


# Parses the text, map unique chars to integers and provides a dictionary to convert char2int and int2char
def Parse(txt):
    chars = sorted(list(set(txt)))
    char2int = dict((c, i) for i, c in enumerate(chars))
    int2char = dict((i, c) for i, c in enumerate(chars))
    
    nChars = len(txt)
    nAlpha = len(chars)
    convDict = {"char2int": char2int, "int2char": int2char}
    return convDict, nChars, nAlpha


# Preprocess for LSTM (seqSize = sentence size)
def Preprocess(txt, seqSize, nChars, nAlpha, char2int):
    dataX = []
    dataY = []
    for i in range(0, nChars - seqSize, 1):
        seqIn = txt[i:i + seqSize]
        seqOut = txt[i + seqSize]
        dataX.append([char2int[char] for char in seqIn])
        dataY.append(char2int[seqOut])
    nPats = len(dataX)
    print ("Total Patterns: ", nPats)
   
    X = numpy.reshape(dataX, (nPats, seqSize, 1))
    # normalize by alphabet size
    X = X / float(nAlpha)
    # Convert dataY to binary class matrix
    y = to_categorical(dataY)
    return X, y , dataX
