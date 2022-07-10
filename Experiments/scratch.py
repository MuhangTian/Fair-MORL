import numpy as np
from Fair_Taxi_MDP import Fair_Taxi_MDP

'''Test MDP environment with mannual actions'''
size = 10
location_coordinates = [[1,1],[3,3],[9,9]]
destination_coordinates = [[1,2],[0,0],[8,9]]
dest_coord = [[1,2]]
env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=dest_coord,
                    fuel=26, fps=2, output_path='Taxi_MDP/run_')
rate = 1
state = env.reset([1,4])
actions = [1,2,2,4,3,3,3,3,1,1,1,5,0,2,4,0,5,1,4,0,5,1,4,0,5,1]
actions_2 =  [1,2,2,4,3,3,3,3,1,1,1,0,2,4,0,5,1,4,0,5,1,4,0,5,1,5]
done = False
for action in actions_2:
    if done == True:
        break
    obs, reward, done, info = env.step(action)
    env.render()