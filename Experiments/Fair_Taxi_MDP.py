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
    
    def __init__(self, size, loc_coords, dest_coords, fuel, output_path) -> None:
        super().__init__()
        self.output_path = output_path
        self.size = size    # size of grid world NxN
        self.window_size = 512
        self.loc_coords = loc_coords    # array of tuples of coordinates for all locations
        self.dest_coords = dest_coords  # array of tuples of coordinates for all destinations
        self.share_dest = True if len(dest_coords) == 1 else False
        self.taxi_loc = None
        self.pass_loc = 0   # 1 is in taxi
        self.pass_dest = None   # destination of the passenger in taxi
        self.pass_idx = None    # to keep track of index of location of the passenger
        self.old_fuel = fuel
        self.current_fuel = fuel
        
        self.acc_reward = np.zeros(len(loc_coords))     # accumulated reward for each location
        self.pass_delivered_loc = np.zeros(len(loc_coords))      # record number of success deliveries for each location
        self.pass_delivered = 0     # record number of total successful deliveries
        self.timesteps = 0
        self.metrics = []       # used to record values
        self.csv_num = 0
        
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
        self.timesteps = 0
        self.current_fuel = self.old_fuel
        state = self.encode(self.taxi_loc[0], self.taxi_loc[1], self.pass_loc)
        
        return state
    
    def _get_info(self):
        dict = {'Taxi Location' : self.taxi_loc, 'Accumulated Reward': self.acc_reward,
                'Fuel Left' : self.current_fuel, 'Passengers Delivered' : self.pass_delivered,
                'Passengers Deliverd by Location' : self.pass_delivered_loc}
        return dict
    
    def _update_metrics(self):
        """
        Update and record performance statistics for each time step
        """
        arr = np.hstack(([self.timesteps], self.acc_reward, self.pass_delivered_loc, [self.pass_delivered]))
        self.metrics.append(arr)
    
    def _produce_labels(self):
        """
        Produce labels for columns in csv file
        """
        labels = ['Timesteps']
        for i in range(len(self.loc_coords)):
            labels.append('Accumulated Reward at Location {}'.format(i))
        for i in range(len(self.loc_coords)):
            labels.append('Delivered Passengers at Location {}'.format(i))
        labels.append('Total Number of Delivered Passengers')
        return labels
    
    def _output_csv(self):
        self.csv_num += 1
        labels = self._produce_labels()
        df = pd.DataFrame(data=self.metrics, columns=labels)
        df.to_csv('{}{}.csv'.format(self.output_path, self.csv_num))
        return 
        
    def step(self, action):
        if action < 0: raise Exception('Invalid Action')
        try: action_mapped = self._action_to_direct[action]
        except: raise Exception('Invalid Action')
        
        if action_mapped == 'pick':
            if self.taxi_loc in self.loc_coords: # TODO: Bug in the condition, related with numpy array stuff
                self.pass_loc = 1   # Passenger now in taxi
                self.pass_idx = self.loc_coords.index(self.taxi_loc) # find destination
                self.pass_dest = self.dest_coords[self.pass_idx]
            reward = np.zeros(len(self.loc_coords))     # NOTE: zero rewards for both valid and invalid pick?
        elif action_mapped == 'drop':
            if self.taxi_loc == self.pass_dest: # TODO: Bug in the condition, related with numpy array stuff
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
            self.taxi_loc = np.where(self.taxi_loc < 0, 0, self.taxi_loc)
            self.taxi_loc = np.where(self.taxi_loc > self.size-1, self.size-1, self.taxi_loc)
            reward = np.zeros(len(self.loc_coords))
        
        self.timesteps += 1
        self.current_fuel -= 1
        self.acc_reward += reward
        
        done = True if self.current_fuel == 0 else False  # terminal state, when fuel runs out
        obs = self.encode(self.taxi_loc[0], self.taxi_loc[1], self.pass_loc)    # next state
        info = self._get_info()
        self._update_metrics()
        if done == True: self._output_csv()
                
        return obs, reward, done, info
    
    def generate_reward(self):  # generate reward based on traveled distance, with a floor of 0
        reward = np.zeros(len(self.loc_coords))
        reward[self.pass_idx] = 30     # dimension of the origin location receive reward
        return reward
        
    def encode(self, taxi_x, taxi_y, pass_loc):
        """
        Encode state with taxi and passenger location into unique integer, used to index Q-table
        """
        code = np.ravel_multi_index([taxi_x, taxi_y, pass_loc], (self.size, self.size, 2))
        return code
    
    # TODO: Test all methods above, debug if necessary
    def render(self, mode='human'):
        # TODO: Finish this method after testing all above methods
        return
        

if __name__ == "__main__":
    loc_coords = [[0,1],[0,2],[0,3]]
    dest_coords = [[1,1],[1,2],[1,3]]
    size = 5
    env = Fair_Taxi_MDP(size=size, loc_coords=loc_coords, dest_coords=dest_coords,
                        fuel=10, output_path='Taxi_MDP/run_')
    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    arr = [1,2,3]
    print(arr.index(1))
        
    
        
        