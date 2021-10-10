"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class RL_nomodel():
    def __init__(self):
            self.env = gym.make('CartPole-v1')
            self.exprnce = list()
            self.hist = list()
            self.actn_space = np.array([0, 1])
            self.exploration_threshold = 0.2
            # NN input is the current observation and action
            # NN output is the reward associated
            self.policynetwork = tf.keras.models.Sequential([
                tf.keras.layers.Dense(32, input_dim=5, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)])
            self.policynetwork.compile(optimizer='adam',
                                       loss=tf.keras.losses.MeanSquaredError())

    def mdl_train(self):
        x_train = list()
        y_train = list()
        for episode in self.exprnce:
            start_obs, actn, _ , _ = episode[0]
            reward = np.sum([step[2] for step in episode])
            x_train.append(np.reshape(np.append(start_obs, actn), [1, 5]))
            y_train.append(reward)
        self.policynetwork.fit(np.array([x_train]), np.array([y_train]), epochs=10, verbose=0)

    def actn_predict_explr(self, observation):
        return (self.actn_space[np.argmax([
                self.policynetwork.predict(np.reshape(np.append(observation, 0), [1, 5])),
                self.policynetwork.predict(np.reshape(np.append(observation, 1), [1, 5]))
                ])]
                if random.random() > self.exploration_threshold
                else random.randint(0,1))

    def actn_predict(self, observation):
        return self.actn_space[np.argmax([
                self.policynetwork.predict(np.reshape(np.append(observation, 0), [1, 5])),
                self.policynetwork.predict(np.reshape(np.append(observation, 1), [1, 5]))
                ])]

    def run(self, plcy_iter, games):
        for i in range(plcy_iter):
            self.exprnce = list()
            for j in range(games):
                done = False
                observation = self.env.reset()
                self.exprnce.append(list())
                tot_reward = 0
                actn = self.actn_predict_explr(observation)    # First action with exploration
                while not done:
                    # self.env.render()
                    newobservation, reward, done, info = self.env.step(actn)
                    self.exprnce[j].append([observation, actn, reward, newobservation])
                    tot_reward += reward
                    observation = newobservation
                    actn = self.actn_predict(observation)    # Subsequent action  have no exploration
                print("[msg] >> Episode terminated with score:", tot_reward)
            self.mdl_train()
            self.hist.append(self.exprnce)

    def close_env(self):
        self.env.close()

    def result(self, iterations):
        avg_reward = [np.mean([len(self.hist[j][i]) for i in range(len(self.hist[j]))]) for j in range(len(self.hist))]
        plt.figure()
        plt.plot(avg_reward)
        plt.xlabel('Policy iterations')
        plt.ylabel('average reward')
        plt.show()
        for i in range(iterations):
            done = False
            observation = self.env.reset()
            tot_reward = 0
            while not done:
                self.env.render()
                actn = self.actn_predict(observation)
                newobservation, reward, done, info = self.env.step(actn)
                tot_reward += reward
                observation = newobservation
            print("[msg] >> Episode terminated with score:", tot_reward)



if __name__ == '__main__':
    myRLmodel = RL_nomodel()
    myRLmodel.run(2,5)
    myRLmodel.result(2)
