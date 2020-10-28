from Utils.Admin.Standard import *
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import Parser as pr


def CalcWeights(txt):
  
    convDict, nChars, nAlpha = pr.Parse(txt)
    char2int = convDict['char2int']

    # prepare the dataset of input to output pairs encoded as integers
    seqSize = 100
    X, y, dataX = pr.Preprocess(txt, seqSize, nChars, nAlpha, char2int)

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # define the checkpoint
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]
    
    # fit the model
    model.fit(X, y, epochs=2, batch_size=128, callbacks=callbacks)
    print("done")

    return model


   
