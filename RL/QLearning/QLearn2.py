import gym,sys,numpy as np
import tensorflow as tf
from gym.envs.registration import register



register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=2000,
    reward_threshold=0.78, # optimum = .8196
)


env = gym.make('FrozenLakeNotSlippery-v0')
Qtable = np.zeros([env.observation_space.n,env.action_space.n])

# Parameters
num_epis = 5000
num_iter = 2000
learning_rate = 0.3
discount = 0.8

# Train
for epis in range(num_epis):
    state = env.reset()
    for iter in range(num_iter):
        action = np.argmax(Qtable[state,:] + np.random.randn(1,4))
        state_new,reward,done,_ = env.step(action)
        Qtable[state,action] = (1-learning_rate)* Qtable[state,action] + \
                                         learning_rate * (reward + discount*np.max(Qtable[state_new,:]) )
        state = state_new
        if done: break
print(np.argmax(Qtable,axis=1))
print(np.around(Qtable,6))


# Plot
s = env.reset()
for _ in range(100):
    action  = np.argmax(Qtable[s,:])
    state_new,_,done,_ = env.step(action)
    env.render()
    s = state_new
    if done: break



env = gym.make('FrozenLake-v0')
env.seed(0)
np.random.seed(56776)
Qtable = np.zeros([env.observation_space.n,env.action_space.n])

# Parameters
num_epis = 500
num_iter = 200
learning_rate = 0.3
discount = 0.8

# Train
for epis in range(num_epis):
    
    state = env.reset()

    for iter in range(num_iter):
        action = np.argmax(Qtable[state,:] + np.random.randn(1,4))
        state_new,reward,done,_ = env.step(action)
        Qtable[state,action] = (1-learning_rate)* Qtable[state,action] + \
                                         learning_rate * (reward + discount*np.max(Qtable[state_new,:]) )
        state = state_new

        if done: break

print(np.argmax(Qtable,axis=1))
print(np.around(Qtable,6))

s = env.reset()
for _ in range(100):
    action  = np.argmax(Qtable[s,:])
    state_new,_,done,_ = env.step(action)
    env.render()
    s = state_new
    if done: break



