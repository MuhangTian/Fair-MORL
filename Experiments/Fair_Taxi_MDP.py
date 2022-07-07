'''
Taxi MDP environment: https://docs.google.com/document/d/16Ot75_Pw65V51QLKaXE1-ifJhQtf4AEkqZ0E87XbKrk/edit
'''
import numpy as np
import pandas as pd
import gym
import pygame
from gym import spaces

class Fair_Taxi_MDP(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, size, loc_coords, dest_coords, fuel) -> None:
        super().__init__()
        self.size = size    # size of grid world NxN
        self.window_size = 512
        self.loc_coords = loc_coords    # array of tuples of coordinates for all locations
        self.dest_coords = dest_coords  # array of tuples of coordinates for all destinations
        self.share_dest = True if len(dest_coords) == 1 else False
        self.taxi_loc = None
        self.pass_loc = 0   # 1 is in taxi
        self.pass_dest = None   # destination of the passenger in taxi
        self.pass_idx = None    # to keep track of index of location of the passenger
        self.fuel = fuel
        self.fuel_spent = 0
        
        self.acc_reward = np.zeros(len(loc_coords))
        self.pass_delivered_loc = np.zeros(len(loc_coords))      # record number of success deliveries for each location
        self.pass_delivered = 0
        self.metrics = []       # used to record values
        
        self.observation_space = spaces.Dict({
            'taxi' : spaces.Box(0, size-1, shape=(2,), dtype=int),  # taxi location
            'passenger' : spaces.Discrete(2),    # 0 for no passenger, 1 for with passenger
        })
        self.observation_space_size = size*size*2
        self.action_space = spaces.Discrete(6)
        self._action_to_direct = {0: np.array([0, -1]),
                                  1: np.array([0, 1]),
                                  2: np.array([1, 0]),
                                  3: np.array([-1, 0]),
                                  4: 'pick', 
                                  5: 'drop'}
        self.window = None
        self.clock = None
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.taxi_loc = self.np_random.integers(0, self.size, size=2)   # random taxi spawn location
        self.pass_loc = 0   # passenger out of taxi
        self.pass_dest = None
        self.pass_idx = None
        self.acc_reward = np.zeros(len(self.loc_coords))
        self.pass_delivered = 0
        self.pass_delivered_loc = np.zeros(len(self.loc_coords))
        state = self.encode(self.taxi_loc[0], self.taxi_loc[1], self.pass_loc)
        
        return state
    
    def _get_info(self):    # NOTE: add more things into the info, that is helpful for generating .csv file
        dict = {'Taxi Location' : self.taxi_loc, 'Accumulated Reward': self.acc_reward,
                'Fuel Left' : self.fuel-self.fuel_spent, 'Passengers Delivered' : self.pass_delivered,
                'Passengers Deliverd by Location' : self.pass_delivered_loc}
        return dict
    
    def _update_metrics(self):
        
    
    def step(self, action):
        try: action_mapped = self._action_to_direct[action]
        except: raise Exception('Invalid Action')
        
        if action_mapped == 'pick':
            if self.taxi_loc in self.loc_coords:
                self.pass_loc = 1   # Passenger now in taxi
                self.pass_idx = self.loc_coords.index(self.taxi_loc) # find destination
                self.pass_dest = self.dest_coords[self.pass_idx]
            reward = np.zeros(len(self.loc_coords))     # NOTE: zero rewards for both valid and invalid pick?
        elif action_mapped == 'drop':
            if self.taxi_loc == self.pass_dest:
                self.pass_loc = 0
                self.pass_dest = None
                self.pass_delivered += 1
                self.pass_delivered_loc[self.pass_idx] += 1
                self.pass_idx = None
                reward = self.generate_reward()
            else:
                reward = np.zeros(len(self.loc_coords))
        else:
            self.taxi_loc += action_mapped  # taxi move according to the map
            reward = np.zeros(len(self.loc_coords))
        
        done = True if self.fuel_spent == self.fuel else False  # terminal state, when fuel run out
        obs = self.encode(self.taxi_loc[0], self.taxi_loc[1], self.pass_loc)    # next state
        info = self._get_info()
        self.acc_reward += reward
        # TODO: write a method that record info and keep track for the csv output at terminal state
        if done == True: self._output_csv()
        
        return obs, reward, done, info
    
    def _output_csv(self):
        # TODO: finsih implementing this method
        return 
    
    def generate_reward(self):  # generate reward based on traveled distance, with a floor of 0
        reward = np.zeros(len(self.loc_coords))
        reward[self.pass_idx] =  30     # dimension of the origin location receive reward
        return reward
        
    def encode(self, taxi_x, taxi_y, pass_loc): # encode observation into an unique integer
        code = np.ravel_multi_index([taxi_x, taxi_y, pass_loc], (self.size, self.size, 2))
        return code
    
    def render(self, mode='human'):
        

if __name__ == "__main__":
    loc_coords = [[0,1],[0,2],[0,3]]
    dest_coords = [[1,1],[1,2],[1,3]]
    size = 1000
    env = Fair_Taxi_MDP(size=size, loc_coords=loc_coords, dest_coords=dest_coords)
    #print(env.reset())
    arr = [1,2,3]
    print(arr.index(1))
        
    
        
        