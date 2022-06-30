'''
Running tests for bandit taxi problem, using our algorithm and Q-learning
'''
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from Fair_Taxi_Bandit import Fair_Taxi_Bandit

def run_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=1):
    Q_table = np.zeros([non_fair_env.observation_space.n, non_fair_env.action_space.n])
    
    for _ in range(episodes):
        state = non_fair_env.reset()
        for _ in range(timesteps):
            if np.random.uniform(0, 1) < epsilon:
                action = non_fair_env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])  # since only one state
            
            next_state, reward, done, info = non_fair_env.step(action)
            reward = np.sum(reward)     # turn vector reward into scalar
            new_value = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[next_state]))
            
            Q_table[state, action] = new_value
            state = next_state
    
    print('FINISH TRAINING Q LEARNING')
    return Q_table

def run_NSW_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=1):
    Q_table = np.ones([fair_env.observation_space.n, fair_env.action_space.n, fair_env.num_locs]) # Cannot use zero due to log
    R_acc = np.zeros(fair_env.num_locs)
    
    for _ in range(episodes):
        state = fair_env.reset()
        for _ in range(timesteps):
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                action = argmax_nsw(R_acc, gamma*Q_table[state], state)
                
            next_state, reward, done, info = fair_env.step(action)
            max_action = argmax_nsw(reward, gamma*Q_table[next_state], next_state)
            new_value = Q_table[state, action] + alpha*(reward + gamma*Q_table[next_state, max_action] - Q_table[state, action])
            
            Q_table[state, action] = new_value
            state = next_state
            R_acc += reward
    
    print('FINISH TRAINING NSW Q LEARNING')
    return Q_table

def nsw(vec): return np.sum(np.log(vec))    # numpy uses natural log

def argmax_nsw(R, gamma_Q, state):
    '''
    Function that finds an action that max NSW(R_acc + gamma*Q_table[state, action])
    :param R: (array) any vector reward to be added
    :param gamma_Q: (2D array) gamma*Q_table[state]
    :param state: (int) current state
    '''
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[state, i]) for i in range(fair_env.action_num)]
    action = np.argmax(nsw_vals)
    return action

def evaluate_Q_learning(Q_table, runs=20, timesteps=10000):
    non_fair_env.clean_all() # clean values before generating results
    state = non_fair_env.reset()
    for i in range(runs):
        for _ in range(timesteps):
            action = np.argmax(Q_table[state])
            next, reward, done, info = non_fair_env.step(action)
        non_fair_env.output_csv()
    return print("FINISH EVALUATE Q LEARNING")

def evaluate_NSW_Q_learning(Q_table, runs=20, timesteps=10000, gamma=1):
    R_acc = np.zeros(fair_env.num_locs)
    fair_env.clean_all()
    state = fair_env.reset()
    for i in range(runs):
        for _ in range(timesteps):
            action = argmax_nsw(R_acc, gamma*Q_table[state], state)
            next, reward, done, info = fair_env.step(action)
            R_acc += reward
        fair_env.output_csv()
    return print("FINSIH EVALUATE NSW Q LEARNING")

if __name__ == "__main__":
    
    non_fair_env = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                       min_mean=10, sd=0.1, 
                       center_mean=20, max_diff=2,
                       output_path='Bandit_Qlearning/Q_learning_run_')

    fair_env = copy.deepcopy(non_fair_env) # copy the same environment, ensure same distribution & randomization
    fair_env.output_path='Bandit_NSW/NSW_Q_learning_run_' # change output path only
    
    Q_table = run_Q_learning()
    nsw_Q_table = run_NSW_Q_learning()
    print('Q learning Q-table:\n{}'.format(Q_table))
    print('NSW Q learning Q-table:\n{}'.format(nsw_Q_table))
    evaluate_Q_learning(Q_table, runs=50, timesteps=100000)
    evaluate_NSW_Q_learning(nsw_Q_table, runs=50, timesteps=100000)