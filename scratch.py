import numpy as np
# from Experiments.Fair_Taxi_MDP import Fair_Taxi_MDP

# size = 5
# loc_coords = [[0,0], [3,2]]
# dest_coords = [[0,4], [3,3]]
# fuel = 1000
# env = Fair_Taxi_MDP(size=size, loc_coords=loc_coords, dest_coords=dest_coords,
#                     fuel=fuel, fps=5, output_path='Taxi_MDP/run_')

# env.reset()
# for _ in range(10000):
#     env.step(env.action_space.sample())
#     env.render()
arr = [1,2,3]
np.save('Test', arr)
