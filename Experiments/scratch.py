import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

# dist = skewnorm.rvs(a=150, loc=10, scale=10, size=1000)
# plt.hist(dist,30,density=True, color = 'b', alpha=0.6)
# plt.title("Reward Distribution")
# plt.show()
# max, min = np.max(dist), np.min(dist)

# print("Max: {}".format(max))
# print("Min: {}".format(min))
# print('Max index: {}'.format(np.where(dist==max)))
# print('Max value using indexing: {}'.format(dist[np.where(dist==max)]))
# print('Min value using indexing: {}'.format(dist[np.where(dist==min)]))
# dist = np.delete(dist, np.where(dist==max))
# dist = np.delete(dist, np.where(dist==min))
# print('New max after deletion: {} \nNew min after deletion: {}'.format(np.max(dist), np.min(dist)))
# print("Random values: {}".format(np.random.choice(dist, replace=False, size=5)))
# print(np.sort([5,4,3,2,1]))