"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time


def stupidpolicy(observation):
    cpos, cvel, pang, pvel = observation
    action = 1 if pvel > 0 else 0
    return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation = env.reset()
    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()     # take a random action
        action = stupidpolicy(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("[msg] >> Episode terminated.")
            time.sleep(2)
            break
    env.close()
