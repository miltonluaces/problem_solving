from Utils.Admin.Standard import *
import sys
import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from Parser import Preprocess


def Generate(txt, model=None, weights=None):

    # Mapping of unprepare_datasetique chars to integers, and reverse
    chars = sorted(list(set(txt)))
    chars.insert(0,'\r')
    print(chars)
    char2int = dict((c, i) for i, c in enumerate(chars))
    int2char = dict((i, c) for i, c in enumerate(chars))

    # summarize
    nChars = len(txt)
    nAlpha = len(chars)
    print("Total Characters: ", nChars)
    print( "Total Alphabet: ", nAlpha)

    # prepare the dataset of input to output pairs encoded as integers
    seqSize = 100
    X, y, dataX = Preprocess(txt, seqSize, nChars, nAlpha, char2int)

    # Create model
    print (X.shape[1], X.shape[2])
    if(model != None):
        model = Sequential()
        model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))

    # Compile
    if(weights != None): model.load_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print ("Seed:")
    print( "\"", ''.join([int2char[value] for value in pattern]), "\"")

    # Generate characters
    for i in range(100):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(nAlpha)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int2char[index]
        seq_in = [int2char[value] for value in pattern]
        #sys.stdout.write(result)
        print(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("\nDone")

