'''
Running tests for bandit taxi problem, using our algorithm and Q-learning
'''
import numpy as np
from Fair_Taxi_Bandit import Fair_Taxi_Bandit

env = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                       min_mean=10, sd=3, 
                       center_mean=20, max_diff=2,
                       output_direct='Bandit_Qlearning/run_')

env2 = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                       min_mean=10, sd=3, 
                       center_mean=20, max_diff=2,
                       output_direct='Bandit_NSW/run_')

def run_Q_learning(timesteps=100000, episodes=1000, alpha=0.1, epsilon=0.1, gamma=0.6):
    Q_table = np.zeros([env.observation_space, env.action_space])
    
    for _ in range(episodes):
        state = env.reset()
        for _ in range(timesteps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])  # since only one state
            
            next_state, reward, done, info = env.step(action)
            reward = np.sum(reward)     # turn vector reward into scalar
            new_value = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[state]))
            Q_table[state, action] = new_value
        
    return Q_table

def run_NSW_Q_learning(timesteps=100000, alpha=0.1, epsilon=0.1, gamma=0.6):
    Q_table = np.zeros([env.observation_space, env.action_space])
    
    for _ in range(timesteps):
        env.reset()
        
        
    return

def visualize():
    return

# env.step(1)
# env.step(1)
# env.step(3)
# print(env._get_info())
# env.reset()
# print(env._get_info())