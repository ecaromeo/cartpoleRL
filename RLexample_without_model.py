"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time
import numpy as np
import tensorflow as tf
import random

class RL_nomodel():
    def __init__(self):
            self.env = gym.make('CartPole-v1')
            self.exprnce = list()
            self.actn_space = np.array([0, 1])
            self.exploration_threshold = 0.1
            # NN input is the current observation
            # NN output is the action probability
            self.policynetwork = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, input_dim=4, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2)])
            self.policynetwork.compile(optimizer='adam',
                                       loss=tf.keras.losses.MeanSquaredError())
            # NN input is the current observation
            # NN output is the value
            self.valuenetwork = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, input_dim=4, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)])
            self.valuenetwork.compile()

    def mdl_train(self):
        x_train = list()
        y_train = list()
        for episode in self.exprnce:
            start_obs, actn, _ , _ = episode[0]
            reward = np.sum([step[2] for step in episode])
            x_train.append(start_obs)
            y_tmp = self.policynetwork.predict(np.reshape(start_obs, [1, 4]))
            y_tmp[0][actn] = reward
            y_train.append(y_tmp[0])
        self.policynetwork.fit(np.array([x_train]), np.array([y_train]), epochs=10)

    def actn_predict(self, observation):
        return (self.actn_space[np.argmax(
                self.policynetwork.predict(np.reshape(observation, [1, 4])))]
                if random.random() > self.exploration_threshold
                else random.randint(0,1))

    def run(self):
        for i in range(20):
            self.exprnce = list()
            for j in range(20):
                done = False
                observation = self.env.reset()
                self.exprnce.append(list())
                tot_reward = 0
                while not done:
                    # self.env.render()
                    actn = self.actn_predict(observation)
                    newobservation, reward, done, info = self.env.step(actn)
                    self.exprnce[j].append([observation, actn, reward, newobservation])
                    tot_reward += reward
                    observation = newobservation
                print("[msg] >> Episode terminated with score:", tot_reward)
            self.mdl_train()
        self.env.close()


if __name__ == '__main__':
    myRLmodel = RL_nomodel()
    myRLmodel.run()
