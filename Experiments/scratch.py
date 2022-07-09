import numpy as np
from Fair_Taxi_MDP import Fair_Taxi_MDP
size = 5
location_coordinates = [[1,1],[3,3]]
destination_coordinates = [[1,2], [0,0]]
env = env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=destination_coordinates,
                        fuel=10000, output_path='Taxi_MDP/run_')
state = env.reset()  # initialize a starting state with random location of taxi, 'state' is encoded integer
done = False
while not done:     # episode ends when fuel runs out
    action = env.action_space.sample()
    next, reward, done, info = env.step(action)
    env.render()
