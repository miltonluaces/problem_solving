import tensorflow as tf
import numpy as np

# Bandit list (thresholds, last is the best)
bandits = [0.2,0,-0.2,-5]
nBandits = len(bandits)

def GetReward(bandit):
    r = np.random.randn(1)
    if r > bandit: return 1
    else: return -1

# Agent : initial weights optimistic (1)
tf.reset_default_graph()
weights = tf.Variable(tf.ones([nBandits])) 
bestAction = tf.argmax(weights)

# Train: feed the reward and best action into the network to compute the loss, and use it to update the network.
rewardVar = tf.placeholder(shape=[1],dtype=tf.float32)
actionVar = tf.placeholder(shape=[1],dtype=tf.int32)
responsibleWeight = tf.slice(weights,actionVar,[1])
loss = -(tf.log(responsibleWeight) * rewardVar)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

episodes = 1000 
totReward = np.zeros(nBandits) 
epsilon = 0.1 #(chance of taking a random action)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < episodes:
        if np.random.rand(1) < epsilon: action = np.random.randint(nBandits)
        else: action = sess.run(bestAction)
        reward = GetReward(bandits[action]) 
        _,resp,ww = sess.run([update,responsibleWeight,weights], feed_dict={rewardVar:[reward],actionVar:[action]})
        
        #Update our running tally of scores.
        totReward[action] += reward
        if i % 50 == 0: print( "Running reward for the " + str(nBandits) + " bandits: " + str(totReward))
        i+=1

# Result
print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)): print("...and it was right!")
else: print("...and it was wrong!")