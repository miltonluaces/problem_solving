import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
sent1 = ['cold', 'cold', 'warm', 'cold']
sent2 = ['hot', 'hot', 'warm', 'cold', 'warm', 'hot']
sent3 = ['cold', 'warm','warm', 'hot']
sent4 = ['warm', 'hot']
doc = [sent1, sent2, sent3, sent4]
print('\nSentences')
print(doc)
words = list(np.concatenate(doc))
print('\nWord vector')
print(words)
 
# integer encode
print('\nInt vector')
le = LabelEncoder()
int_vec = le.fit_transform(words)
print(int_vec)

# binary encode
ohe = OneHotEncoder(sparse=False, categories='auto')
int_vec = int_vec.reshape(len(int_vec), 1)
oh_vec = ohe.fit_transform(int_vec)
print('\nOne hot vector')
print(oh_vec)

# invert first example
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#print('\nInverted')
#print(inverted)
