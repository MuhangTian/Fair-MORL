'''
Simplified fair taxi environment: https://docs.google.com/document/d/1NvALGY0kPwjz6ibj9H4S5taTDGVZWC3pKA1PXseI8r4/edit
'''
import gym
import numpy as np
from gym import spaces
from gym.envs.toy_text import discrete

class Fair_Taxi_Simplified(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self, num_locs, output_direct) -> None:
        self.num_locs = num_locs    # number of locations
        self.action_num = num_locs
        self.output_direct = output_direct
        
        self.action_space = spaces.Discrete(self.action_num)
        self.observation_space = spaces.Discrete(1)     # we only have one state due to simplification
        
    def 
        
        