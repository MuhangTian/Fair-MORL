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
        self.loc_coords = np.array(loc_coords)
        self.dest_coords = np.array(dest_coords)  
        self.share_dest = True if len(dest_coords) == 1 else False
        self.taxi_loc = None
        self.pass_loc = 0   # 1 is in taxi
        self.pass_dest = None   # destination of the passenger in taxi
        self.pass_idx = None    # to keep track of index of location of the passenger
        self.fuel = fuel
        
        self.acc_reward = np.zeros(len(loc_coords))     # accumulated reward for each location
        self.pass_delivered_loc = np.zeros(len(loc_coords))      # record number of success deliveries for each location
        self.pass_delivered = 0     # record number of total successful deliveries
        self.timesteps = 0
        self.metrics = []       # used to record values
        self.csv_num = 0
        
        self.observation_space = spaces.Discrete(size*size*2)
        self.action_space = spaces.Discrete(6)
        self._action_to_direct = {0: np.array([0, -1]),
                                  1: np.array([0, 1]),
                                  2: np.array([1, 0]),
                                  3: np.array([-1, 0])}
        self.window = None
        self.clock = None

    def _clean_metrics(self): 
        '''
        clean accumulated reward and past data
        '''
        self.metrics = []
        self.acc_reward = np.zeros(len(self.loc_coords))
        self.pass_delivered_loc = np.zeros(len(loc_coords))
        self.pass_delivered = 0
        
    
    def reset(self, seed=None):
        '''
        Initialize random state to begin with
        '''
        super().reset(seed=seed)
        self.taxi_loc = self.np_random.integers(0, self.size, size=2)   # random taxi spawn location
        self.pass_loc = 0   # passenger out of taxi
        self.pass_dest = None
        self.pass_idx = None
        self.pass_delivered = 0
        self.pass_delivered_loc = np.zeros(len(self.loc_coords))
        self.timesteps = 0
        state = self.encode(self.taxi_loc[0], self.taxi_loc[1], self.pass_loc)
        
        return state
    
    def _get_info(self):
        dict = {'Taxi Location' : self.taxi_loc, 'Accumulated Reward': self.acc_reward,
                'Fuel Left' : self.fuel-self.timesteps, 'Passengers Delivered' : self.pass_delivered,
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
            labels.append('Location {} Accumulated Reward'.format(i))
        for i in range(len(self.loc_coords)):
            labels.append('Location {} Delivered Passengers'.format(i))
        labels.append('Total Delivered Passengers')
        return labels
    
    def _output_csv(self):
        self.csv_num += 1
        labels = self._produce_labels()
        df = pd.DataFrame(data=self.metrics, columns=labels)
        df.to_csv('{}{}.csv'.format(self.output_path, self.csv_num))
        return 
        
    def step(self, action):
        if action < 0 or action > 5: raise Exception('Invalid Action')
        
        if action == 4: # pick
            if self.taxi_loc.tolist() in self.loc_coords.tolist() and self.pass_loc == 0:
                self.pass_loc = 1   # Passenger now in taxi
                self.pass_idx = self.loc_coords.tolist().index(self.taxi_loc.tolist()) # find destination
                self.pass_dest = self.dest_coords[self.pass_idx]
            
            reward = np.zeros(len(self.loc_coords))     # NOTE: zero rewards for both valid and invalid pick?
        elif action == 5:   # drop
            if np.array_equal(self.taxi_loc, self.pass_dest) and self.pass_loc == 1: 
                self.pass_loc = 0
                self.pass_dest = None
                self.pass_delivered += 1
                self.pass_delivered_loc[self.pass_idx] += 1
                self.pass_idx = None
                reward = self.generate_reward()
            else:
                self.pass_loc = 0
                self.pass_dest = None
                self.pass_idx = None
                reward = np.zeros(len(self.loc_coords))
        else:
            self.taxi_loc += self._action_to_direct[action]  # taxi move according to the map
            self.taxi_loc = np.where(self.taxi_loc < 0, 0, self.taxi_loc)
            self.taxi_loc = np.where(self.taxi_loc > self.size-1, self.size-1, self.taxi_loc)
            reward = np.zeros(len(self.loc_coords))
        
        self.timesteps += 1
        self.acc_reward += reward
        
        done = True if self.timesteps == self.fuel else False  # terminal state, when fuel runs out
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
    
    def render(self, mode='human'):
        '''
        Create  graphics
        '''
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels
        
        font = pygame.font.SysFont('Times New Roman', 30, bold=False)
        
        # Draw Locations (blue)
        loc_pos = []
        for i in range(len(self.loc_coords)):
            loc = pygame.Rect(pix_square_size * self.loc_coords[i], (pix_square_size, pix_square_size),)
            pygame.draw.rect(
                surface = canvas,
                color = (28,134,238),
                rect = loc,
            )
            label = font.render(str(i), True, (0,0,0))
            label_rect = label.get_rect()
            label_rect.center = loc.center
            loc_pos.append([label, label_rect])
        
        # Draw Destinations (red)
        dest_pos = []
        for i in range(len(self.dest_coords)):
            loc = pygame.Rect(pix_square_size * self.dest_coords[i], (pix_square_size, pix_square_size),)
            pygame.draw.rect(
                surface = canvas,
                color = (238,64,0),
                rect = loc,
            )
            label = font.render(str(i), True, (0,0,0))
            label_rect = label.get_rect()
            label_rect.center = loc.center
            dest_pos.append([label, label_rect])

        # Draw Agent (yellow when no passenger, green when there is passenger with passenger index)
        if self.pass_idx == None:
            agent = pygame.draw.circle(
                    surface = canvas,
                    color = (255,185,15),
                    center = (self.taxi_loc + 0.5) * pix_square_size,
                    radius = pix_square_size / 3,
                    ) 
        else:
            agent = pygame.draw.circle(
                    surface = canvas,
                    color = (0,205,0),
                    center = (self.taxi_loc + 0.5) * pix_square_size,
                    radius = pix_square_size / 3,
                    ) 
            pass_label = font.render(str(self.pass_idx), True, (0,0,0))
            pass_label_rect = pass_label.get_rect()
            pass_label_rect.center = agent.center
        
        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        
        # Output visualization
        self.window.blit(canvas, canvas.get_rect())
        if self.pass_idx != None: self.window.blit(pass_label, pass_label_rect)
        for i in range(len(self.loc_coords)): self.window.blit(loc_pos[i][0], loc_pos[i][1])
        for i in range(len(self.loc_coords)): self.window.blit(dest_pos[i][0], dest_pos[i][1])
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"]) # add a delay to keep the framerate stable
        return
        

if __name__ == "__main__":
    # For testing
    loc_coords = [[8,1],[3,8],[5,5],[4,4]]
    dest_coords = [[6,1],[0,9],[0,0],[1,1]]
    size = 10
    env = Fair_Taxi_MDP(size=size, loc_coords=loc_coords, dest_coords=dest_coords,
                        fuel=10000, output_path='Taxi_MDP/run_')
    
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next, reward, done, info = env.step(action)
        env.step(4)
        env.render()
        
    
        
        