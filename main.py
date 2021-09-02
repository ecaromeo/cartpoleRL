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
    cost = 1/(0.418 - np.abs(pang)) + 1/(4.8 - np.abs(cpos))
    return cost


def calc_CostToGo(observation, iterations):
    """
    Recursively calculate the cost to go
    """
    action_space = np.array([0, 1])
    new_states = [apprx_calcNewState(observation,
                                     action) for action in action_space]
    immediate_costs = [apprx_costFunction(state) for state in new_states]

    if iterations == 0:
        return np.min(immediate_costs)
    else:
        return np.min([immed_cost + calc_CostToGo(newstate, iterations - 1)
                       for immed_cost, newstate in 
                       zip(immediate_costs, new_states)])


def apprx_calcNewState(observation, action):
    """
    Calculate new state, given the current state
    and the action.
    @param observation Is an array with four elements
    describing the current state
    @param action Takes value in the set [0, 1]
    """
    cpos, cvel, pang, pvel = observation
    new_cpos = cpos + 0.1*cvel
    if np.abs(new_cpos) > 4.8:
        new_cpos = np.sign(new_cpos)*4.8
    new_cvel = cvel + 0.1*(2 * action - 1)
    new_pang = pang + 0.1*pvel
    if np.abs(new_pang) > 0.418:
        new_pang = np.sign(new_pang)*0.418
    new_pvel = pvel + 0.1*(- 2 * action + 1)
    new_observation = np.array([new_cpos, new_cvel,
                                new_pang, new_pvel])
    return new_observation


def plcy_1(observation):
    """
    Policy 1.
    """
    action_space = np.array([0, 1])
    new_states = [apprx_calcNewState(observation,
                                     action) for action in action_space]
    immediate_cost = [apprx_costFunction(state) for state in new_states]
    cost_to_go = [calc_CostToGo(state, 5) for state in new_states]
    slctd = np.argmin(np.add(immediate_cost, cost_to_go))

    return action_space[slctd]


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation = env.reset()
    tot_reward = 0
    while True:
        env.render()
        # action = env.action_space.sample()     # take a random action
        # action = stupidpolicy(observation)
        action = plcy_1(observation)
        observation, reward, done, info = env.step(action)
        tot_reward += reward
        if done:
            print("[msg] >> Episode terminated with score:", tot_reward)
            time.sleep(2)
            break
    env.close()
