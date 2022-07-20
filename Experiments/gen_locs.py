'''Generate random locations for taxi environment'''
import numpy as np
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def random_locs(num_locs, size):
    arr = []
    count = 0
    while count != num_locs:
        loc = np.random.random_integers(0, size-1, size=2)
        loc = loc.tolist()
        if loc not in arr:
            arr.append(loc)
            count += 1
        else: continue
    return arr

def random_dests(num_dests, size, loc_coords):
    arr = []
    count = 0
    while count != num_dests:
        loc = np.random.random_integers(0, size-1, size=2)
        loc = loc.tolist()
        if (loc not in arr) and (loc not in loc_coords): 
            arr.append(loc)
            count += 1
        else: continue
    return arr

if __name__ == '__main__':
    size = 7
    num_locs = 6
    loc_coords = random_locs(num_locs, size)
    dest_coords = random_dests(num_locs, size, loc_coords)
    np.save('Experiments/rand_locs/locs_size{}_num{}.npy'.format(size, num_locs), loc_coords)
    np.save('Experiments/rand_locs/dests_size{}_num{}.npy'.format(size, num_locs), dest_coords)
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, 10000, '', fps=1)
    for _ in range(10000):
        env.reset()
        env.render()