"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    experience = list()
    done = False
    # NN input is the current observation
    # NN output is the action probability
    policynetwork = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=4, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)])
    policynetwork.compile()
    # NN input is the current observation
    # NN output is the action probability
    valuenetwork = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=4, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])
    valuenetwork.compile()

    for i in range(1000):
        observation = env.reset()
        tot_reward = 0
        while not done:
            env.render()
            actn = env.action_space.sample()    # take a random action
            # actn = stupidpolicy(observation)    # applies a simple policy
            newobservation, reward, done, info = env.step(actn)
            experience.append([observation, actn, reward, newobservation])
            tot_reward += reward
            observation = newobservation
        print("[msg] >> Episode terminated with score:", tot_reward)
    env.close()
