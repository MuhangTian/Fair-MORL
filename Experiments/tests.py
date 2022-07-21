import numpy as np
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from Fair_Taxi_MDP import Fair_Taxi_MDP

'''Test MDP environment with mannual actions'''
# size = 10
# location_coordinates = [[1,1],[3,3],[9,9]]
# destination_coordinates = [[1,2],[0,0],[8,9]]
# dest_coord = [[1,2]]
# env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=destination_coordinates,
#                     fuel=26, fps=2, output_path='Taxi_MDP/run_')
# rate = 1
# state = env.reset([1,4])
# actions = [1,2,2,4,3,3,3,3,1,1,1,5,0,2,4,0,5,1,4,0,5,1,4,0,5,1]
# actions_2 =  [1,2,2,4,3,3,3,3,1,1,1,0,2,4,0,5,1,4,0,5,1,4,0,5,1,5]
# done = False
# for action in actions_2:
#     if done == True:
#         break
#     obs, reward, done, info = env.step(action)
#     env.render()

'''Test encode method and total possible states'''
# size = 100
# location_coordinates = np.arange(16)
# destination_coordinates = np.arange(16)
# env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=destination_coordinates,
#                     fuel=26, fps=2, output_path='Taxi_MDP/run_')

# arr = []
# for i in range(size):
#     for j in range(size):
#         for k in range(2):
#             for z in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,None]:
#                 arr.append(env.encode(i,j,k,z))
# print(len(arr))
# print(len(set(arr)))
# print(env.observation_space.n)

# code = env.encode(12,99,1,16)
# print(env.decode(code))

'''Test number of reachable states'''
# size = 6
# location_coordinates = [[1,1], [2,2], [3,3], [4,4]]
# destination_coordinates = [[1,2], [2,3], [3,4], [4,1]]
# env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=destination_coordinates,
#                     fuel=26, fps=2, output_path='Taxi_MDP/run_')
# arr = []
# state = env.reset()
# arr.append(state)
# for _ in range(1000000):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     arr.append(obs)
# print(len(set(arr)))

'''Test taxi environment with penalty'''
size = 8
loc_coords = [[0,0], [0,5], [3,2], [3,7], [5,6]]
dest_coords = [[0,4], [5,0], [3,3], [7,0], [6,6]]
fuel = 100

# size = 5
# loc_coords = [[0,0], [3,2]]
# dest_coords = [[0,4], [3,3]]

env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, 
                    output_path='Taxi_MDP/NSW_Q_learning/run_', fps=1)
env.seed(1122)
arr= []    
for _ in range(1000):
    state = env.reset()
    env.render()



