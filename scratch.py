from matplotlib.pyplot import fill
from Experiments.Fair_Taxi_Bandit import Fair_Taxi_Bandit
import numpy as np

# env = Fair_Taxi_Bandit(num_locs=5, max_mean=100, 
#                                     min_mean=10, sd=0, 
#                                     center_mean=20, max_diff=0,
#                                     output_path='Bandit_Qlearning/Q_learning_run_')
# env.step(1)
# print(env.rewards)
# env.step(2)
# print(env.rewards)
# print(env.rewards)
# env.step(1)
# print(env.rewards)

arr = [[1,2,3], [3,2,1], [4,5,6]]
time = np.arange(1,4)
print(np.sum(arr, axis=0))
print(np.divide(arr, time))
