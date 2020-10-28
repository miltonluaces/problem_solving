# Implementing an RNN in Tensorflow
#----------------------------------
#
# We implement an RNN in Tensorflow to predict spam/ham from texts
#

import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile


# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 20
bSize = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)


# Download or open data
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii',errors='ignore')
    text_data = text_data.decode().split('\n')

    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.80)
Xtrain, Xtest = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
Ytrain, Ytest = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(Ytrain), len(Ytest)))

# Create placeholders
Xdata = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, Xdata)
#embedding_output_expanded = tf.expand_dims(embedding_output, -1)


# Define the RNN cell
cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_out, y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

trainLoss = []
testLoss = []
trainAcc = []
testAcc = []
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(Xtrain)))
    Xtrain = Xtrain[shuffled_ix]
    Ytrain = Ytrain[shuffled_ix]
    num_batches = int(len(Xtrain)/bSize) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * bSize
        max_ix = np.min([len(Xtrain), ((i+1) * bSize)])
        Xtrain_batch = Xtrain[min_ix:max_ix]
        Ytrain_batch = Ytrain[min_ix:max_ix]
        
        # Run train step
        train_dict = {Xdata: Xtrain_batch, y_output: Ytrain_batch, dropout_keep_prob:0.5}
        sess.run(step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_trainLoss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    trainLoss.append(temp_trainLoss)
    trainAcc.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {Xdata: Xtest, y_output: Ytest, dropout_keep_prob:1.0}
    temp_testLoss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    testLoss.append(temp_testLoss)
    testAcc.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_testLoss, temp_test_acc))
    
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, trainLoss, 'k--', label='Train Set')
plt.plot(epoch_seq, testLoss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, trainAcc, 'k--', label='Train Set')
plt.plot(epoch_seq, testAcc, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()