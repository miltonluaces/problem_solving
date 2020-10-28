import numpy as np
import tensorflow as tf

class VAE(object):
    
    def __init__(self, nnArch, transfer=tf.nn.softplus, lrate=0.001, bSize=100):
        self.nnArch = nnArch
        self.transfer = transfer
        self.lrate = lrate
        self.bSize = bSize
        self.x = tf.placeholder(tf.float32, [None, nnArch["nInput"]]) 
        
        self.createNN()
        self.createLossOpt() 

        init = tf.global_variables_initializer() 
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def createNN(self):
        weights = self.initWeights(**self.nnArch)
        self.zMu, self.zLogS2 = self.recNN(weights["Wrec"], weights["brec"]) 
        nZ = self.nnArch["nZ"] # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.bSize, nZ), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.zMu, tf.multiply(tf.sqrt(tf.exp(self.zLogS2)), eps)) # z = mu + sigma*epsilon
        self.xRecMean = self.genNN(weights["Wgen"], weights["bgen"]) # Use generator to determine mean of Bernoulli distribution of reconstructed input
            
    def xInit(self, fan_in, fan_out, constant=1): 
        low = -constant * np.sqrt(6.0/(fan_in + fan_out)) 
        high = constant * np.sqrt(6.0/(fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    def initWeights(self, nHrec1, nHrec2, nHgen1,  nHgen2, nInput, nZ):
        weights = dict()
        weights['Wrec'] = {
            'h1': tf.Variable(self.xInit(nInput, nHrec1)),
            'h2': tf.Variable(self.xInit(nHrec1, nHrec2)),
            'outMean': tf.Variable(self.xInit(nHrec2, nZ)),
            'outLogS': tf.Variable(self.xInit(nHrec2, nZ))}
        weights['brec'] = {
            'b1': tf.Variable(tf.zeros([nHrec1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([nHrec2], dtype=tf.float32)),
            'outMean': tf.Variable(tf.zeros([nZ], dtype=tf.float32)),
            'outLogS': tf.Variable(tf.zeros([nZ], dtype=tf.float32))}
        weights['Wgen'] = {
            'h1': tf.Variable(self.xInit(nZ, nHgen1)),
            'h2': tf.Variable(self.xInit(nHgen1, nHgen2)),
            'outMean': tf.Variable(self.xInit(nHgen2, nInput)),
            'outLogS': tf.Variable(self.xInit(nHgen2, nInput))}
        weights['bgen'] = {
            'b1': tf.Variable(tf.zeros([nHgen1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([nHgen2], dtype=tf.float32)),
            'outMean': tf.Variable(tf.zeros([nInput], dtype=tf.float32)),
            'outLogS': tf.Variable(tf.zeros([nInput], dtype=tf.float32))}
        return weights
            
    # Generate probabilistic encoder (recognition network), which maps inputs onto a normal distribution in latent space. The transformation is parametrized and can be learned. Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
    def recNN(self, weights, biases):
        layer1 = self.transfer(tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])) 
        layer2 = self.transfer(tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])) 
        zMu = tf.add(tf.matmul(layer2, weights['outMean']), biases['outMean'])
        zLogS2 = tf.add(tf.matmul(layer2, weights['outLogS']), biases['outLogS'])
        return (zMu, zLogS2)

    # Generate probabilistic decoder (decoder network), which maps points in latent space onto a Bernoulli distribution in data space. The transformation is parametrized and can be learned.
    def genNN(self, weights, biases):
        layer1 = self.transfer(tf.add(tf.matmul(self.z, weights['h1']), biases['b1'])) 
        layer2 = self.transfer(tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])) 
        xRecMean = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['outMean']), biases['outMean']))
        return xRecMean
            
    # Define loss function based variational upper-bound and corresponding optimizer. The loss is composed of two terms:
    # 1 Reconstruction loss (negative log probability of the input under the reconstructed Bernoulli distribution induced by the decoder in the data space) i.e. the number of "nats" required for reconstructing the input when  activation in latent is given
    # 2.Latent loss: the Kullback Leibler divergence between the distribution in latent space induced by the encoder on the data and some prior. This acts as a kind of regularizer. This can be interpreted as the number of "nats" required for transmitting the the latent space distribution given the prior.
    def createLossOpt(self):
        recLoss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.xRecMean) + (1-self.x) * tf.log(1e-10 + 1 - self.xRecMean),1)
        latLoss = -0.5 * tf.reduce_sum(1 + self.zLogS2 - tf.square(self.zMu) - tf.exp(self.zLogS2), 1)
        self.cost = tf.reduce_mean(recLoss + latLoss)  
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate).minimize(self.cost)
        
    #Train model based on mini-batch of input data. Return cost of batch
    def Fit(self, X):
        return self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
    
    # Transform data by mapping it into the latent space
    def transform(self, X):
        return self.sess.run(self.zMu, feed_dict={self.x: X})
    
    # Generate data by sampling from latent space
    def generate(self, zMu):
        return self.sess.run(self.xRecMean, feed_dict={self.z: zMu})
    
    # Reconstruct original data
    def reconstruct(self, X):
        return self.sess.run(self.xRecMean, feed_dict={self.x: X})

    def train(self, dataset, epochs):
        printStep = 5
        for epoch in range(epochs):
            cost = self.Fit(dataset)
            if epoch % printStep == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost))  
        

