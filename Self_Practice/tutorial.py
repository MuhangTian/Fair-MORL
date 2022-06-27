import gym
import matplotlib.pyplot as plt
import time 
from collections import deque
from gym import spaces
import numpy as np

env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')

# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)

obs = env.reset()

class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)

def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
        self.frames.append(ob)
    return self._get_ob()

def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

def _get_ob(self):
    return np.array(self.frames)

wrapped_env = ConcatObs(env, 4)
print("The new observation space is", wrapped_env.observation_space)

# for i in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     #env.render()
#     time.sleep(0.01)
# env.close()
