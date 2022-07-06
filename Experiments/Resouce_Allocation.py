'''
MDP version of bandit environment: https://docs.google.com/document/d/1a967zAXDXZdUMKqzGVzJlEbP_LgteCiHiXC7S5eB8gI/edit
'''
import numpy as np
import gym
from gym import spaces

class Resouce_Allocation(gym.Env):
    def __init__(self, num_users, goods_dist, path, manual_utils) -> None:
        self.num_users = num_users
        self.num_goods = len(goods_dist)    # number of types of goods
        self.path = path
        self.users_utils = np.array(manual_utils)
        self.goods_dist = goods_dist
        self.initial_goods_dist = goods_dist
        self.goods_left = np.sum(goods_dist)    # episode terminates after this number reaches 0
        self.current_state = None
        
        self.action_space = spaces.Discrete(num_users)
        self.observation_space = spaces.Discrete(self.num_goods)
        
        self.metrics = []
        self.accum_reward = np.zeros(num_users, dtype=float)
        
    def _generate_state(self, goods_dist):  # generate state (give a good) according to goods distribution
        goods = np.arange(0, self.num_goods)
        goods_dist = goods_dist / np.sum(goods_dist)    # create probability for each good
        state = np.random.choice(goods, p=goods_dist)
        return state
    
    def _get_info(self):
        return 'Goods left: {}'.format(self.goods_left)
    
    def reset(self):
        self.goods_left = np.sum(self.initial_goods_dist)
        self.goods_dist = self.initial_goods_dist   # restore good distribution to original
        state = self._generate_state(self.initial_goods_dist)
        self.current_state = state
        return state
    
    def step(self, action): # give a good to a particular user & update goods distribution
        if action < 0: raise Exception('Invalid Action')
        try:
            user = self.users_utils[action]
        except:
            raise Exception('Invalid Action')
        
        reward = user[self.current_state]     # user's utility for the particular good given
        self.goods_dist[self.current_state] -= 1    # good given out, one less good
        self.goods_left = np.sum(self.goods_dist)
        observation = self._generate_state(self.goods_dist)
        
        if self.goods_left == 0:
            done = True
        else:
            done = False
        info = self._get_info()
        return observation, reward, done, info
        
if __name__ == "__main__":
    dist = [10, 20, 30, 40]
    env = Resouce_Allocation(num_users=5, goods_dist=dist, path='', manual_utils=[])
    print(env.observation_space.sample())