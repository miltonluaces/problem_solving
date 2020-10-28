from Utils.Admin.Standard import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy
import spacy
sy = spacy.load('en')
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle


# Methods

def CreateModel(seqLength, vocabSize):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(lstmSize, activation="relu"),input_shape=(seqLength, vocabSize)))
    model.add(Dropout(0.6))
    model.add(Dense(vocabSize))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=m)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built")
    return model

# Sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Parameters
vocabFile = os.path.join(modelPath, "wordsVocab.pkl")
seqStep = 1 #step to create sequences


# 1. Read data


txtFile = txtPath + 'wonderlandSample.txt'
txt = open(txtFile).read()
doc = sy(txt)

wordlist = []
for word in doc:
    if word.text not in ("\n","\n\n",'\u2009','\xa0'): wordlist.append(word.text.lower())





# 2. Create the dictionary

# Word count
wc = collections.Counter(wordlist)

# Mapping idx2word
vocabInv = [x[0] for x in wc.most_common()]
vocabInv = list(sorted(vocabInv))

# Mapping word2idx
vocab = {x: i for i, x in enumerate(vocabInv)}
words = [x[0] for x in wc.most_common()]

#size of the vocabulary
vocabSize = len(words)
print("vocab size: ", vocabSize)

#save the words and vocabulary
with open(os.path.join(vocabFile), 'wb') as f:
    cPickle.dump((words, vocab, vocabInv), f)


# 3 Create Sentences List
lstmSize = 256
seqLength = 30 
m = 0.001

sequences = []
nextWords = []
for i in range(0, len(wordlist) - seqLength, seqStep):
    sequences.append(wordlist[i: i + seqLength])
    nextWords.append(wordlist[i + seqLength])

print('nb sequences:', len(sequences))

X = np.zeros((len(sequences), seqLength, vocabSize), dtype=np.bool)
y = np.zeros((len(sequences), vocabSize), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence): X[i, t, vocab[word]] = 1
    y[i, vocab[nextWords[i]]] = 1


# 4. Build the Bidirectional LSTM Model
md = CreateModel(seqLength, vocabSize)
md.summary()


# 5 Train and save the Model
batchSize = 32 # minibatch size
epochs = 2 # number of epochs

callbacks=[EarlyStopping(patience=4, monitor='val_loss'), ModelCheckpoint(filepath=modelPath + "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, mode='auto', period=2)]
history = md.fit(X, y, batch_size=batchSize, shuffle=True, epochs=epochs, callbacks=callbacks, validation_split=0.1)
md.save(modelPath + "/" + 'biLstmModel.h5')


# 6 Generate Sentences
print("loading vocabulary...")
vocabFile = os.path.join(modelPath, "wordsVocab.pkl")

with open(os.path.join(modelPath, 'wordsVocab.pkl'), 'rb') as f:
        words, vocab, vocabInv = cPickle.load(f)

vocabSize = len(words)

print("loading model...")
model = load_model(modelPath + "/" + 'biLstmModel.h5')

words_number = 30 # number of words to generate
seed_sentences = "alice was beginning to get very tired of sitting by her sister" #seed sentence to start the generating.

#initiate sentences
generated = ''
sentence = []

#we shate the seed accordingly to the neural netwrok needs:
for i in range (seqLength): sentence.append("a")

seed = seed_sentences.split()
for i in range(len(seed)):
    sentence[seqLength-i-1]=seed[len(seed)-i-1]

generated += ' '.join(sentence)

#the, we generate the text
for i in range(words_number):
    #create the vector
    x = np.zeros((1, seqLength, vocabSize))
    for t, word in enumerate(sentence): x[0, t, vocab[word]] = 1.

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = vocabInv[next_index]

    #add the next word to the text
    generated += " " + next_word
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

#print the whole text
print(generated)
