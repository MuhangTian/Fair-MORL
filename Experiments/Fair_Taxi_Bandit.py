'''
Simplified fair taxi environment: https://docs.google.com/document/d/1NvALGY0kPwjz6ibj9H4S5taTDGVZWC3pKA1PXseI8r4/edit

Fair bandit taxi problem:
1. One state only, taxi driver choose which location to pick from in each timestep
2. Environment gives a vector reward associated with the agent's action of choice

Something special:
1. Can set number of locations arbitrarily, num_locs = num_actions in this environment
2. Can set reward generation and distributions arbitrarily, allow us to adjust environments when testing
3. When call output_csv(), generates a table of timesteps, and accumulated reward so far at each location
at every time step, this makes plotting the graphs easier.
4. Rewards for each location is sorted, 1st location is smallest, Nth location is largest
'''
import gym
import numpy as np
import pandas as pd
from gym import spaces
class Fair_Taxi_Bandit(gym.Env):
    '''
    :param int num_locs: number of locations to choose from
    :param int max_mean: maximum mean for all distributions
    :param int min_mean: minimum mean for all distributions
    :param int sd: the shared, constant standard deviation for all location's distributions
    :param int center_mean: the mean value which most distributions' means are around (except max and min)
    :param int max_diff: the maximum difference from center_mean allowed for variation in means (except max and min)
    :param str output_path: output directory of accumulated reward .csv file
    '''
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self, num_locs=5, max_mean=40, min_mean=10, sd=3, center_mean=20, max_diff=2, output_path='Bandit_Fair_Taxi_Run') -> None:
        self.num_locs = num_locs    # number of locations
        self.max_mean = max_mean
        self.min_mean = min_mean
        self.sd = sd
        self.center_mean = center_mean
        self.max_diff = max_diff
        self.action_num = num_locs
        self.output_path = output_path
        
        self.action_space = spaces.Discrete(self.action_num)
        self.observation_space = spaces.Discrete(1)     # only one state due to simplification
        self.accum_reward = np.zeros(num_locs, dtype=float)
        self.rewards = None
        self.metrics = []
        self.timesteps = 0
        self.output_count = 0

    def _get_info(self):
        return  'Time step: {}\nAccumulated reward: {}'.format(self.timesteps, self.accum_reward)
    
    def reset(self):
        return 0    # since only one state, return 0

    def clean_all(self):
        self.accum_reward = np.zeros(self.num_locs, dtype=float)
        self.metrics = []
        self.timesteps = 0
        return
    
    def output_csv(self):
        self.output_count += 1
        df = pd.DataFrame(data=self.metrics, columns=self.produce_labels())
        df.to_csv(self.output_path+'{}.csv'.format(self.output_count))
        return

    def update_metrics(self):
        arr = np.append([self.timesteps], self.accum_reward)
        self.metrics.append(arr)
        return
    
    def produce_labels(self):
        labels = ['Time steps']
        for i in range(self.num_locs):
            labels.append('Location {}'.format(i))
        return labels
    
    def step(self, action):
        if action < 0:
            raise Exception('Invalid Action')
        
        try:
            self.rewards = self.generate_vec_reward(self.max_mean, self.min_mean, 
                                                    self.sd, self.center_mean, 
                                                    self.max_diff)  # random reward for each episode
            reward = self.rewards[action]
            self.accum_reward += reward
        except:
            raise Exception('Invalid Action')
        
        observation = 0     # Since only have one state
        done = False        # Bandit problem, no terminal state
        info = self._get_info()
        self.timesteps += 1
        self.update_metrics()
        
        return observation, reward, done, info
        
    def generate_vec_reward(self, max_mean, min_mean, sd, center_mean, max_diff):
        '''
        Generate vector reward for each location, stored as 2D array
        '''
        rewards, result = np.array([], dtype=float), np.array([], dtype=float)
        for i in range(self.num_locs):
            if i == 0:
                rewards = np.append(rewards, np.random.normal(loc=max_mean, scale=sd))
            elif i == self.num_locs-1:
                rewards = np.append(rewards, np.random.normal(loc=min_mean, scale=sd))
            else:
                rand = np.random.uniform(low=-max_diff, high=max_diff)
                rewards = np.append(rewards, np.random.normal(loc=center_mean+rand, scale=sd))       
        rewards = np.sort(rewards)  # Sort to ensure rewards can be compared between experiments
        
        for i in range(self.num_locs):   # create 2D reward array
            zero = np.zeros(self.num_locs, dtype=float)
            if i == 0:
                zero[i] = rewards[i]
                result = np.append(result, zero)
            else:
                zero[i] = rewards[i]
                result = np.vstack((result, zero))
                
        return result
    
    