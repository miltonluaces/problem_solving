# Data gathering
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import requests
from tensorflow.examples.tutorials.mnist import input_data
import io
from zipfile import ZipFile
import tarfile

# Iris data
iris = datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.data[0])
print(set(iris.target))

# Low Birthrate Data
url = 'https://raw.githubusercontent.com/h2oai/h2o-2/master/smalldata/logreg/umass_statdata/lowbwt.dat'
file = requests.get(url)
print(file.text)
data = [item.split() for item in file.text.split('\n')[5:]]
print(len(data))
print(len(data[0]))
print(data[:3])

# Housing Price Data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
file = requests.get(url)
data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in file.text.split('\n') if len(y)>=1]
print(len(data))
print(len(data[0]))


# MNIST Handwriting Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images))
print(len(mnist.test.images))
print(len(mnist.validation.images))
print(mnist.train.labels[1,:])


# Ham/Spam Text Data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
file = requests.get(url)
zFile = ZipFile(io.BytesIO(file.content))
file = zFile.read('SMSSpamCollection')
# Format Data
data = file.decode()
data = data.encode('ascii',errors='ignore')
data = data.decode().split('\n')
data = [x.split('\t') for x in data if len(x)>=1]
[dataTarget, dataTrain] = [list(x) for x in zip(*data)]
print(len(dataTrain))
print(set(dataTarget))
print(dataTrain[1])


# Movie Review Data
url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
file = requests.get(url)
# Stream data into temp object
stream = io.BytesIO(file.content)
tmp = io.BytesIO()
while True:
    s = stream.read(16384)
    if not s:  
        break
    tmp.write(s)
stream.close()
tmp.seek(0)

# Extract tar file
tarFile = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tarFile.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tarFile.extractfile('rt-polaritydata/rt-polarity.neg')
# Save pos/neg reviews
posData = []
for line in pos:
    posData.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
negData = []
for line in neg:
    negData.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
tarFile.close()

print(len(posData))
print(len(negData))
print(negData[0])


# The Works of Shakespeare Data
url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(url)
file = response.content
# Decode binary into string
text = file.decode('utf-8')
# Drop first few descriptive paragraphs.
text = text[7675:]
print(len(text))


# English-German Sentence Translation Data
url = 'http://www.manythings.org/anki/deu-eng.zip'
file = requests.get(url)
zFile = ZipFile(io.BytesIO(file.content))
file = zFile.read('deu.txt')

# Format Data
data = file.decode()
data = data.encode('ascii',errors='ignore')
data = data.decode().split('\n')
data = [x.split('\t') for x in data if len(x)>=1]
[english, german] = [list(x) for x in zip(*data)]
print(len(english))
print(len(german))
print(data[10])
