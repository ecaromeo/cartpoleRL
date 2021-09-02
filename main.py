"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time
import numpy as np


def stupidpolicy(observation):
    cpos, cvel, pang, pvel = observation
    action = 1 if pvel > 0 else 0
    return action

def apprx_costFunction(observation):
    cpos, cvel, pang, pvel = observation
    cost = np.abs(pang) + np.abs(cpos)
    return cost

def calcNewState(observation, action):
    """
    Calculate new state, given the current state
    and the action.
    @param observation Is an array with four elements
    describing the current state
    @param action Takes value in the set [0, 1]
    """
    cpos, cvel, pang, pvel = observation
    new_cpos = cpos + cvel
    new_cvel = cvel + 0.1*(2 * action - 1)
    new_pang = pang + pvel
    new_pvel = pvel + 0.1*(- 2 * action + 1)
    new_observation = np.array([new_cpos, new_cvel,
                                new_pang, new_pvel])
    return new_observation



def plcy_1(observation):
    """
    Policy 1. 
    """
    action_space = np.array([0, 1])
    slctd = np.argmin([apprx_costFunction(
                       calcNewState(observation,
                                    action))
                       for action in action_space])
    return action_space[slctd]

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation = env.reset()
    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()     # take a random action
        # action = stupidpolicy(observation)
        action = plcy_1(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("[msg] >> Episode terminated.")
            time.sleep(2)
            break
    env.close()
