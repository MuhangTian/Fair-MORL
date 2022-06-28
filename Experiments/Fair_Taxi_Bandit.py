'''
Simplified fair taxi environment: https://docs.google.com/document/d/1NvALGY0kPwjz6ibj9H4S5taTDGVZWC3pKA1PXseI8r4/edit
'''
import gym
import numpy as np
from gym import spaces
from gym.envs.toy_text import discrete
import scipy.stats as ss
'''
To do:
1. Write Gaussian distributions for each possible location, with a sensible difference in means
2. 
'''
class Fair_Taxi_Bandit(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self, num_locs, output_direct) -> None:
        self.num_locs = num_locs    # number of locations
        self.action_num = num_locs
        self.output_direct = output_direct
        
        self.action_space = spaces.Discrete(self.action_num)
        self.observation_space = spaces.Discrete(1)     # only one state due to simplification
        self.rewards = self.generate_vec_reward(skewness=150, loc=10, scale=10, size=1000) # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
        self.action_reward_map = self.action_to_reward()
    
    def _get_info(self):
        
    def step(self, action):
        try:
            reward = self.action_reward_map[action]
        except:
            raise Exception('Invalid Action')
        observation = 1     # Since only have one state
        done = False        # Bandit problem, no terminal state
        
        return observation, reward, done, info
        
    def generate_vec_reward(self, skewness, loc, scale, size):
        '''
        Use skewed distribution to generate rewards, return a 2D array of rewards for choosing each location
        '''
        dist = ss.skewnorm.rvs(a=skewness, loc=loc, scale=scale, size=size)
        max, min = np.max(dist), np.min(dist)      # use one max and one min from distribution for two locations
        max_idx, min_idx = np.where(dist==max), np.where(dist==min)
        
        dist = np.delete(dist, max_idx)     # Delete max and min to avoid sampling with replacement
        dist = np.delete(dist, min_idx)
        
        other_rewards = np.random.choice(dist, replace=False, size=self.num_locs-2) # Generate scalar rewards for remaining locations
        other_rewards = np.sort(other_rewards)
        
        result = np.array([], dtype=float)      # creat output
        for i in range(self.num_locs):
            zero = np.zeros(self.num_locs, dtype=float)
            if i == 0:
                zero[i] = min
                result = np.append(result, zero)
            elif i == self.num_locs-1:
                zero[i] = max
                result = np.vstack((result, zero))
            else:
                zero[i] = other_rewards[i-1]
                result = np.vstack((result, zero))
        
        return result
    
    def action_to_reward(self): # return dictionary of actions paired with reward of choosing that action
        map = {}
        for i in range(self.num_locs):
            map[i] = self.rewards[i]
        return map
        

if __name__ == '__main__':  # For testing
    n = 6
    env = Fair_Taxi_Bandit(n, 'output_path')
    print(env.rewards) # reward for choosing each of N locations
    print(env.action_reward_map)
    