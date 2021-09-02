"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time
import numpy as np


def stupidpolicy(observation):
    """
    This function implements a very simple policy
    which pushes the cart in the direction opposite
    to the pole rotation.

    @param observation      Is an array with four elements
    describing the current state. In order:
    (i) cart position; (ii) cart velocity;
    (iii) pole angle (iv) pole angular velocity
    """
    cpos, cvel, pang, pvel = observation
    action = 1 if pvel > 0 else 0
    return action


def apprx_costFunction(observation):
    """
    The function provides a simple cost approximation.
    Cost is computed as a non-linear function of the
    distance to the limits of the pole angle and of the
    cart position.

    @param observation      Is an array with four elements
    describing the current state. In order:
    (i) cart position; (ii) cart velocity;
    (iii) pole angle (iv) pole angular velocity
    """
    cpos, cvel, pang, pvel = observation
    cost = 1/(0.418 - np.abs(pang)) + 1/(4.8 - np.abs(cpos))
    return cost


def calc_CostToGo(observation, iterations):
    """
    Recursively calculate the cost to go starting from a given state.
    The number of iterations is fixed by the iterations parameter.

    @param observation      Is an array with four elements
    describing the current state. In order:
    (i) cart position; (ii) cart velocity;
    (iii) pole angle (iv) pole angular velocity

    @param iterations       The number of recursive iterations
    this can be thought as the depth of the lookahead tree
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
    and the action. The calculation is approximate

    @param observation Is an array with four elements
    describing the current state. In order:
    (i) cart position; (ii) cart velocity;
    (iii) pole angle (iv) pole angular velocity

    @param action Takes value in the set [0, 1]
    """
    cpos, cvel, pang, pvel = observation
    new_cpos = cpos + 0.1*cvel
    if np.abs(new_cpos) > 4.8:
        # Do now allow cart positions beyond +/-4.8
        new_cpos = np.sign(new_cpos)*4.8
    new_cvel = cvel + 0.1*(2 * action - 1)
    new_pang = pang + 0.1*pvel
    if np.abs(new_pang) > 0.418:
        # Do not allow pole angles beyond +/-0.418
        new_pang = np.sign(new_pang)*0.418
    new_pvel = pvel + 0.1*(- 2 * action + 1)
    new_observation = np.array([new_cpos, new_cvel,
                                new_pang, new_pvel])
    return new_observation


def plcy_1(observation):
    """
    Implements the policy. For each possible action calculates the
    immediate costs and cost_to_go. Then selects the action with minimum
    cost

    @param observation      Is an array with four elements
    describing the current state. In order:
    (i) cart position; (ii) cart velocity;
    (iii) pole angle (iv) pole angular velocity
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
        # actn = env.action_space.sample()    # take a random action
        # actn = stupidpolicy(observation)    # applies a simple policy
        actn = plcy_1(observation)            # applies a policy with lookahead
        observation, reward, done, info = env.step(actn)
        tot_reward += reward
        if done:
            print("[msg] >> Episode terminated with score:", tot_reward)
            time.sleep(2)
            break
    env.close()
