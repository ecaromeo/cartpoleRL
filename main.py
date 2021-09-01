"""
Runs a Reinforcement Learning simulation.

Author: Romeo Casesa
"""

import gym
import time

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()     # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("[msg] >> Episode terminated.")
            time.sleep(2)
            break
    env.close()
