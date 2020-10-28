import numpy as np
import pandas as pd 
import math 
import tensorflow as tf 

class AE(object):

    # Fields
    nHidden = 2
    epochs = 1000
    batchSize = 10
    Whid = None
    bhid = None
    Hout = None
    Wout = None
    bout = None
    
    Yout = None
    Yexp = None
    Out = None
    inputOrig = None
    input = None
    output = None
    nSamp = 0
    nVar = 3
    X = tf.placeholder("float", [None, nVar]) 
 
    
    
    # Constructor
    def __init__(self):
        return
      
    # Normalize to [0,1] 
    def Norm01(self, data):
        nData = np.divide((data-data.min()), (data.max() - data.min())) 
        return nData
    
    # Normalize to [-1,1] 
    def Norm_11(self, data):
        nData = np.divide((data-data.min()), (data.max() - data.min())) 
        n1Data = (nData*2)-1 
        return n1Data

    def Build(self, input): 
         
        # Normalization (in this case [-1,1]
        self.inputOrig = input
        output = input 
        self.input = self.Norm_11(input) 
        self.output = self.Norm_11(output) 
         
        self.nSamp, self.nVar = self.input.shape  
        self.batchSize = min(self.batchSize, self.nSamp) 
        
        # input layer
        self.X = tf.placeholder("float", [None, self.nVar]) 
       
        # hidden layer 
        self.Whid = tf.Variable(tf.random_uniform((self.nVar, self.nHidden), -1.0 / math.sqrt(self.nVar), 1.0 / math.sqrt(self.nVar))) 
        self.bhid = tf.Variable(tf.zeros([self.nHidden])) 
        self.Yhid = tf.nn.tanh(tf.matmul(self.X,self.Whid) + self.bhid) 
        
        # output layer
        self.Wout = tf.transpose(self.Whid) # tied weights 
        self.bout = tf.Variable(tf.zeros([self.nVar])) 
        self.Yout = tf.nn.tanh(tf.matmul(self.Yhid,self.Wout) + self.bout) 
        
        # expected outcome
        self.Yexp = tf.placeholder("float", [None,self.nVar]) 
        return
    
    def Train(self, log=False): 
        
        mse = tf.reduce_mean(tf.square(self.Yout - self.Yexp)) #crossEntropy = -tf.reduce_sum(self.Yexp * tf.log(self.Yout)) 
        trainStep = tf.train.GradientDescentOptimizer(lrate=0.05).minimize(mse) 
        
        init = tf.global_variables_initializer()
        sess = tf.Session() 
        sess.run(init) 
   
        for i in range(self.epochs): 
            sample = np.random.randint(self.nSamp, size=self.batchSize) 
            bX = self.input[sample][:] 
            bY = self.output[sample][:] 
            sess.run(trainStep, feed_dict={self.X: bX, self.Yexp:bY}) 
            if i % 100 == 0 and log==True: 
                Mse = sess.run(mse, feed_dict={self.X: bX, self.Yexp:bY}); print(i, " : ", round(Mse,5)) 
        self.Out = sess.run(self.Yout, feed_dict={self.X: self.input})
        return
    
    def Output(self):
        #print("Expected:"); print(self.output) 
        #print( "Output:"); print(np.round(self.Out, 2)); 
        diff = np.square(self.output - self.Out); 
        M = np.mean(pd.DataFrame(diff), axis=1); 
        Mm = M.as_matrix().reshape(M.shape[0], 1)
        res = np.append(self.inputOrig, Mm, axis=1); print(res)
        return res
