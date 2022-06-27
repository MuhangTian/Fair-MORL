'''
@author: MuhangTian
Mutli-objective taxi environment, an adjustment over Gym's Taxi environment, idea in this link: 
https://docs.google.com/document/d/1zwqsvJEyCIKmGWJkQq666vjmYEDjudkmW5HSVasDTZ4/edit
'''
import gym
import numpy as np
import sys

from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete


MAP = [
    "+---------+",
    "|R: |W| :G|",
    "| : | | : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

arr = []   # For testing

class FairTaxi(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")
        self.locs = locs = [(0, 0), (0, 2), (0, 4), (4, 0), (4, 3)]     # store location coordinates
        
        num_states = 715625     # see calculation in Google doc
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.dict_initial_state_distrib = dict_initial_state_distrib = {}     # use dict first then copy into array because cannot get perfect encode function
        initial_state_distrib = np.array([])    
        num_actions = 6
        
        self.P = P = {}  # Transition Matrix, P(s',r|s,a)
        
        # Search through every possible states and next states, and fill transition matrix
        for row in range(num_rows):
            for col in range(num_columns):
                for pass1_idx in range(len(locs)+2): # add two since can be in taxi, or out of the game
                    for pass2_idx in range(len(locs)+2):    # pass_idx: 5 = in taxi, 6 = delivered
                        for pass3_idx in range(len(locs)+2):
                            for dest1_idx in range(len(locs)):
                                for dest2_idx in range(len(locs)):
                                    for dest3_idx in range(len(locs)):
                                        
                                        if self.valid_locs(pass1_idx, pass2_idx, pass3_idx) == False:
                                            continue    # Check validity of passenger locations, can't be at the same place
                                        
                                        state = self.encode(row, col, pass1_idx, pass2_idx, pass3_idx, dest1_idx, dest2_idx, dest3_idx)
                                        P[state] = {action: [] for action in range(num_actions)}    # Initialize transition matrix
                                        # Check whether current state can be a valid initial state (not in taxi, not already in destination, not already delivered)
                                        if (pass1_idx <= 4 and pass1_idx != dest1_idx) and (pass2_idx <= 4 and pass2_idx != dest2_idx) and (pass3_idx <= 4 and pass3_idx != dest3_idx):
                                            self.update_state_distrib(state, dict_initial_state_distrib) 
                                               
                                        for action in range(num_actions):
                                            # defaults
                                            new_row, new_col, new_pass1_idx, new_pass2_idx, new_pass3_idx = row, col, pass1_idx, pass2_idx, pass3_idx
                                            vec_reward = np.array([-1,-1,-1])   # default reward when there is no pickup/dropoff
                                            done = False
                                            taxi_loc = (row, col)    
                                            
                                            # state transitions given current action and state
                                            if action == 0:
                                                new_row = min(row+1, max_row)
                                            elif action == 1:
                                                new_row = max(row-1, 0)
                                            elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                                new_col = min(col + 1, max_col) # move right, check there is road ":"
                                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                                new_col = max(col - 1, 0)
                                            elif action == 4:   # Pick up
                                                # Check if any of the passengers satisfy requirement
                                                # not in taxi, taxi in same place as passnger location
                                                # update passenger index for that passenger
                                                # else, illegal action, reduce reward by [-5,-5,-5]???
                                                vec_reward = np.array([-5,-5,-5])
                                                
                                            
                                                       
                                        
                                        
                                        
                                        
                                        
                                    
                                    
    def valid_locs(self, idx1, idx2, idx3):
        if idx1 == idx2 == idx3 == 6 or idx1 == idx2 == 6 or idx1 == idx3 == 6 or idx2 == idx3 == 6:
            return True     # case for 2 or 3 delivered passengers
        elif (idx1 != idx2) and (idx1 != idx3) and (idx2 != idx3):
            # case for 1 delivered passenger, and count other possible locations
            return True
        else:
            return False
    
    def update_state_distrib(self, state, distrib):
        try:
            distrib[state] += 1
        except:
            distrib[state] = 1
    
    def find_valid_pickup(self, pass1_idx, pass2_idx, pass3_idx, taxi_loc):
        # check if any passenger can be picked up, return passenger number, else return false
        if pass1_idx <= 4 and taxi_loc == self.locs[pass1_idx]:
            return 1
        elif pass2_idx <= 4 and taxi_loc == self.locs[pass2_idx]:
            return 2
        elif pass3_idx <= 4 and taxi_loc == self.locs[pass3_idx]:
            return 3
        else:
            return False
    
    def encode(self, taxi_row, taxi_col, pass1_idx, pass2_idx, pass3_idx, dest1_idx, dest2_idx, dest3_idx):
        # Encode state into an unique number to represent state
        code = np.ravel_multi_index([taxi_row, taxi_col, pass1_idx, pass2_idx, pass3_idx, dest1_idx, dest2_idx, dest3_idx],
                            (5,5,7,7,7,5,5,5))
        return code
    
    def decode(self, code):
        return np.unravel_index(code, (5,5,7,7,7,5,5,5))                       

'''This is for testing'''
if __name__ == "__main__":
    print(FairTaxi().P)            
        
        
        
