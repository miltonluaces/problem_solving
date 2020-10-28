import unittest
import gym
from DPG.policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np


class TestQLearning(unittest.TestCase):

    

    def test01_Acrobot(self):
        env = gym.make('Acrobot-v1')
        env = env.unwrapped
        env.seed(1)

        print("env.action_space", env.action_space)
        print("env.observation_space", env.observation_space)
        print("env.observation_space.high", env.observation_space.high)
        print("env.observation_space.low", env.observation_space.low)


        RENDER_ENV = False
        EPISODES = 500
        rewards = []
        RENDER_REWARD_MIN = -500
        PG = PolicyGradient(n_x = env.observation_space.shape[0], n_y = env.action_space.n, learning_rate=0.02, reward_decay=0.99)

        for episode in range(EPISODES):
            observation = env.reset()
            episode_reward = 0
            while True:
                if RENDER_ENV: env.render()
                action = PG.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                PG.store_transition(observation, action, reward)
                if done:
                    episode_rewards_sum = sum(PG.episode_rewards)
                    rewards.append(episode_rewards_sum)
                    max_reward_so_far = np.amax(rewards)

                    print("\nEpisode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)

                    discounted_episode_rewards_norm = PG.learn()
                    if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                    break
                observation = observation_


    def test02_CarPole(self):
        env = gym.make('CartPole-v0')
        env = env.unwrapped
        env.seed(1)

        print("env.action_space", env.action_space)
        print("env.observation_space", env.observation_space)
        print("env.observation_space.high", env.observation_space.high)
        print("env.observation_space.low", env.observation_space.low)

        RENDER_ENV = False
        EPISODES = 500
        rewards = []
        RENDER_REWARD_MIN = 50

        # Load checkpoint
        load_path = None #"output/weights/CartPole-v0.ckpt"
        save_path = None #"output/weights/CartPole-v0-temp.ckpt"

        PG = PolicyGradient(n_x = env.observation_space.shape[0], n_y = env.action_space.n, learning_rate=0.01,
            reward_decay=0.95, load_path=load_path, save_path=save_path)

        for episode in range(EPISODES):
            observation = env.reset()
            episode_reward = 0
            while True:
                if RENDER_ENV: env.render()
                action = PG.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                PG.store_transition(observation, action, reward)
                if done:
                    episode_rewards_sum = sum(PG.episode_rewards)
                    rewards.append(episode_rewards_sum)
                    max_reward_so_far = np.amax(rewards)

                    print("\nEpisode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)

                    discounted_episode_rewards_norm = PG.learn()
                    if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                    break
                observation = observation_


    def test03_MountainCar(self):
        env = gym.make('MountainCar-v0')
        env = env.unwrapped
        env.seed(1)

        print("env.action_space", env.action_space)
        print("env.observation_space", env.observation_space)
        print("env.observation_space.high", env.observation_space.high)
        print("env.observation_space.low", env.observation_space.low)

        RENDER_ENV = True
        EPISODES = 500
        rewards = []
        RENDER_REWARD_MIN = -1000

        PG = PolicyGradient(n_x = env.observation_space.shape[0],n_y = env.action_space.n,learning_rate=0.02,reward_decay=0.99)

        for episode in range(EPISODES):
            observation = env.reset()
            episode_reward = 0
            while True:
                if RENDER_ENV: env.render()
                action = PG.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                PG.store_transition(observation, action, reward)
                if done:
                    episode_rewards_sum = sum(PG.episode_rewards)
                    rewards.append(episode_rewards_sum)
                    max_reward_so_far = np.amax(rewards)

                    print("\nEpisode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)

                    discounted_episode_rewards_norm = PG.learn()
                    if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                    break
                observation = observation_
