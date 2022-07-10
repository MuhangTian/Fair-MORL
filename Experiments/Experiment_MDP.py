import numpy as np
from Fair_Taxi_MDP import Fair_Taxi_MDP

if __name__ == "__main__":
    size = 10
    location_coordinates = [[1,1],[3,3], [9,9]]
    destination_coordinates = [[1,2], [0,0], [8,9]]
    env = Fair_Taxi_MDP(size=size, loc_coords=location_coordinates, dest_coords=destination_coordinates,
                        fuel=26, fps=2, output_path='Taxi_MDP/run_')
    rate = 1
    state = env.reset([1,4])
    for _ in range(rate): env.render()
    env.step(1)
    for _ in range(rate): env.render()
    env.step(2)
    for _ in range(rate): env.render()
    env.step(2)
    for _ in range(rate): env.render()
    env.step(4)
    for _ in range(rate): env.render()
    env.step(3)
    for _ in range(rate): env.render()
    env.step(3)
    for _ in range(rate): env.render()
    env.step(3)
    for _ in range(rate): env.render()
    env.step(3)
    for _ in range(rate): env.render()
    env.step(1)
    for _ in range(rate): env.render()
    env.step(1)
    for _ in range(rate): env.render()
    obs, reward, done, info = env.step(1)
    for _ in range(rate): env.render()
    obs, reward, done ,info = env.step(5)
    for _ in range(rate): env.render()
    env.step(0)
    for _ in range(rate): env.render()
    env.step(2)
    for _ in range(rate): env.render()
    
    while not done:
        env.step(4)
        env.render()
        env.step(0)
        env.render()
        obs, reward, done, info = env.step(5)
        env.render()
        obs, reward, done, info = env.step(1)
        env.render()
        
    
    