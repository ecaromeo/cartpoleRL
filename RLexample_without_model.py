"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from keras.models import load_model


class RL_nomodel():
    def __init__(self):
        """!
        Initializes the RL class.
        """
        self.env = gym.make('CartPole-v1')
        self.exprnce = list()
        self.hist = list()
        self.actn_space = np.array([0, 1])
        # NN input is the current observation and action
        # NN output is the reward associated
        try:    # Load neural network if present..
            self.policynetwork = load_model("policyNet")
        except OSError:     # ..otherwise create a new one.
            self.policynetwork = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, input_dim=5, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)])
            self.policynetwork.compile(
                optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError())

    def mdl_train(self):
        """!
        Function to train the Policy Neural Network
        based on the last experience. Only the latest
        trajectories are used.
        """
        x_train = list()
        y_train = list()
        for episode in self.exprnce:    # Loop through different trajectories
            final_reward = np.sum([step[2] for step in episode])
            # For each trajectory loop through the individual steps
            for step, i in zip(episode, range(len(episode))):
                x_train.append(np.reshape(
                    np.append(self.nrmlize(step[0]), step[1]), 5))
                y_train.append(np.reshape(
                    (final_reward - i)/1000, 1))
        # Train policyNet with new data
        self.policynetwork.fit(np.array(x_train), np.array(y_train),
                               epochs=100, verbose=0)

    def nrmlize(self, obs):
        """!
        Normalize the state variables for feeding into the NN.
        """
        cpos, cvel, pang, pvel = obs
        return [cpos/4.8, cvel/100, pang/0.418, pvel/100]

    def actn_predict_explr(self, observation, exploration_threshold):
        """!
        Run the Policy Neural Network to obtain the action to be taken.
        Action is selected as the one with the highest reward based on the
        input state.
        """
        return (self.actn_space[np.argmax([
                self.policynetwork.predict(
                    np.reshape(np.append(observation, 0), [1, 5])),
                self.policynetwork.predict(
                    np.reshape(np.append(observation, 1), [1, 5]))
                ])]
                if random.random() > exploration_threshold
                else random.randint(0, 1))

    def train(self, plcy_iter, games):
        """!
        Train the model.
        @param plcy_iter Number of policy iterations
        @param games Number of trajectories for each policy iteration
        """
        for i in range(plcy_iter):
            self.exprnce = list()
            # Trajectories (or games) upon which the policy is evaluated
            # This data will be used for training the policyNet
            for j in range(games):
                done = False
                observation = self.env.reset()
                # To increase chanses of starting with a non-favorable
                # condition I take four random steps
                self.env.step(random.randint(0, 1))
                self.env.step(random.randint(0, 1))
                self.env.step(random.randint(0, 1))
                observation, _, _, _ = self.env.step(random.randint(0, 1))

                self.exprnce.append(list())
                tot_reward = 0  # Total reward of trajectory
                while not done:     # While the simulation does not end
                    # apply policy to decide next action
                    actn = self.actn_predict_explr(self.nrmlize(observation),
                                                   0.1)
                    newobservation, reward, done, _ = self.env.step(actn)
                    self.exprnce[j].append([observation, actn,
                                            reward, newobservation])
                    tot_reward += reward
                    observation = newobservation
                print("[msg] >> Policy iteratin,", i+1, "episode", j+1,
                      "terminated with score:", tot_reward)
            if i != (plcy_iter-1):
                self.mdl_train()    # Policy improvement
            self.hist.append(self.exprnce)  # Save the trajectories
            self.policynetwork.save('policyNet')    # Save the Neural Network

    def close_env(self):
        """!
        Close the simulation environment
        """
        self.env.close()

    def run(self, iterations, plot=False):
        """!
        Running a number of simulations.
        If available, there is the possibility to show performance
        improvements over the different policy iterations.

        @params iterations Number of simulations to run and show
        @params plot whether to plot or not the average reward of past
        iterations
        """
        if plot:
            if self.hist:
                # Compute average reward for each iteration
                avg_reward = [np.mean([len(self.hist[j][i])
                                       for i in range(len(self.hist[j]))])
                              for j in range(len(self.hist))]
                # Plotting
                plt.figure()
                plt.plot(avg_reward)
                plt.xlabel('Policy iterations')
                plt.ylabel('average reward')
                plt.grid()
                plt.show()
                input('Enter any key to proceed..')
            else:
                print("No history to use for plotting. Skipping..")

        # Run 'iterations' number of policy iterations
        for i in range(iterations):
            done = False
            observation = self.env.reset()
            tot_reward = 0
            while not done:
                self.env.render()   # Render environment
                actn = self.actn_predict_explr(observation, -1)
                newobservation, reward, done, info = self.env.step(actn)
                tot_reward += reward
                observation = newobservation
            print("[msg] >> Episode terminated with score:", tot_reward)


if __name__ == '__main__':
    myRLmodel = RL_nomodel()    # Initialize class
    # Train the policyNet on XX policy iterations each
    # with YY simulations
    myRLmodel.train(5, 50)
    myRLmodel.run(2, True)      # Run XX simulations and show results
    myRLmodel.close_env()       # Close the simulation engine.
