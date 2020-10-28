# Implementing TF-IDF
#---------------------------------------
#
# Here we implement TF-IDF,
#  (Text Frequency - Inverse Document Frequency)
#  for the spam-ham text data.
#
# We will use a hybrid approach of encoding the texts
#  with sci-kit learn's TFIDF vectorizer.  Then we will
#  use the regular Tensorflow logistic algorithm outline.

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer


# Start a graph session
sess = tf.Session()

bSize = 200
max_features = 1000


# Check if data was downloaded, otherwise download it and save for future use
save_file_name = 'temp_spam_data.csv'
if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii',errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]
    
    # And write to csv
    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)


texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# Relabel 'spam' as 1, 'ham' as 0
target = [1. if x=='spam' else 0. for x in target]

# Normalize text
# Lower case
texts = [x.lower() for x in texts]

# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# Define tokenizer
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

# Split up data set into train/test
trainIdx = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
testIdx = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(trainIdx)))
texts_train = sparse_tfidf_texts[trainIdx]
texts_test = sparse_tfidf_texts[testIdx]
target_train = np.array([x for ix, x in enumerate(target) if ix in trainIdx])
target_test = np.array([x for ix, x in enumerate(target) if ix in testIdx])

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare logistic model (sigmoid in loss function)
res = tf.add(tf.matmul(Xdata, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(res, Ytarget))

# Actual Prediction
prediction = tf.round(tf.sigmoid(res))
predictions_correct = tf.cast(tf.equal(prediction, Ytarget), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.0025)
step = opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
trainLoss = []
testLoss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    index = np.random.choice(texts_train.shape[0], size=bSize)
    x = texts_train[index].todense()
    y = np.transpose([target_train[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    
    # Only record loss and accuracy every 100 generations
    if (i+1)%100==0:
        i_data.append(i+1)
        trainLoss_temp = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
        trainLoss.append(trainLoss_temp)
        
        testLoss_temp = sess.run(loss, feed_dict={Xdata: texts_test.todense(), Ytarget: np.transpose([target_test])})
        testLoss.append(testLoss_temp)
        
        train_acc_temp = sess.run(accuracy, feed_dict={Xdata: x, Ytarget: y})
        train_acc.append(train_acc_temp)
    
        ta = sess.run(accuracy, feed_dict={Xdata: texts_test.todense(), Ytarget: np.transpose([target_test])})
        test_acc.append(ta)
    if (i+1)%500==0:
        acc_and_loss = [i+1, trainLoss_temp, testLoss_temp, train_acc_temp, ta]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


# Plot loss over time
plt.plot(i_data, trainLoss, 'k-', label='Train Loss')
plt.plot(i_data, testLoss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()