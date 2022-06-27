import gym
import numpy as np
import random

env = gym.make("Taxi-v2").env
env.reset()

for i in range(10):
    # action = env.action_space.sample()
    # obs, reward, done, info = env.step(action)
    env.render()
    env.reset()
    