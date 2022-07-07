import numpy as np
import pandas as pd
import gym
from gym import spaces

box = spaces.Box(0, 4, shape=(2,), dtype=int)
print(box)

class env(gym.Env):
    
    def __init__(self) -> None:
        super().__init__()