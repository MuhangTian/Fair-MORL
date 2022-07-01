'''
Running tests for bandit taxi problem, using NSW bandit algorithm and classical bandit algorithm
'''
import numpy as np
import copy
from Fair_Taxi_Bandit import Fair_Taxi_Bandit

def run_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99):
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
            
            print('Accumulated reward so far: {}'.format(non_fair_env.accum_reward))
    
    print('FINISH TRAINING Q LEARNING')
    return Q_table

def run_NSW_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99):
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
            
            print('Accumulated reward so far: {}'.format(R_acc))
    
    print('FINISH TRAINING NSW Q LEARNING')
    return Q_table

def run_bandit_algorithm(timesteps=10000, epsilon=0.1): # P32 Sutton & Barto
    Q_table = np.zeros(non_fair_env.action_space.n)
    num_visited = np.zeros(non_fair_env.action_space.n)
    
    for _ in range(timesteps):
        
        if np.random.uniform(0,1) < epsilon:
            action = non_fair_env.action_space.sample()
        else:
            action = np.argmax(Q_table)
            
        next_state, reward, done, info = non_fair_env.step(action)
        reward = np.sum(reward) # turn in to scalar reward
        num_visited[action] += 1
        Q_table[action] = Q_table[action] + (reward - Q_table[action])/num_visited[action]
        print('Accumualted reward so far: {}'.format(non_fair_env.accum_reward))
    print('FINISH TRAINING BANDIT')
    return Q_table

def run_nsw_bandit_algorithm(timesteps=10000, epsilon=0.1):
    # Q_table = np.full([fair_env.action_space.n, fair_env.action_space.n], fill_value=1e-9)
    Q_table = np.zeros([fair_env.action_space.n, fair_env.action_space.n])
    num_visited = np.zeros(fair_env.action_space.n)
    R_acc = np.zeros(fair_env.action_space.n)
    
    for _ in range(timesteps):
        
        if np.random.uniform(0,1) < epsilon:
            action = fair_env.action_space.sample()
        else:
            action = bandit_argmax_nsw(R_acc, Q_table)
            
        next_state, reward, done, info = fair_env.step(action)
        num_visited[action] += 1
        new_value = Q_table[action] + (reward - Q_table[action])/num_visited[action]
        Q_table[action] = new_value
        R_acc += reward
        print('Accumualted reward so far: {}'.format(fair_env.accum_reward))
    print('FINISH TRAINING NSW BANDIT')
    return Q_table

def nsw(vec): return np.sum(np.log(vec))    # numpy uses natural log

def bandit_argmax_nsw(R, Q):
    '''
    Function that finds an action that max NSW(R_acc + gamma*Q_table[state, action])
    :param R: (array) any vector reward to be added
    :param gamma_Q: (2D array) gamma*Q_table[state]
    :param state: (int) current state
    '''
    sum = R + Q
    nsw_vals = [nsw(sum[i]) for i in range(fair_env.action_num)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = np.random.randint(fair_env.action_num)
    else:
        action = np.argmax(nsw_vals)
        
    return action

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

def evaluate_bandit(Q_table, runs=20, timesteps=10000):
    for i in range(runs):
        non_fair_env.clean_all() # clean values before generating results for each run
        
        for _ in range(timesteps):
            action = np.argmax(Q_table)
            next, reward, done, info = non_fair_env.step(action)
            
        non_fair_env.output_csv()
    return print("FINISH EVALUATE BANDIT")

def evaluate_nsw_bandit(Q_table, runs=20, timesteps=10000):
    for _ in range(runs):
        R_acc = np.zeros(fair_env.num_locs)
        fair_env.clean_all()
        
        for _ in range(timesteps):
            action = bandit_argmax_nsw(R_acc, Q_table)
            next, reward, done, info = fair_env.step(action)
            R_acc += reward
            
        fair_env.output_csv()
    return print("FINSIH EVALUATE NSW BANDIT")

def evaluate_Q_learning(Q_table, runs=20, timesteps=10000):
    for i in range(runs):
        non_fair_env.clean_all() # clean values before generating results for each run
        state = non_fair_env.reset()
        
        for _ in range(timesteps):
            action = np.argmax(Q_table[state])
            next, reward, done, info = non_fair_env.step(action)
            
        non_fair_env.output_csv()
    return print("FINISH EVALUATE Q LEARNING")

def evaluate_NSW_Q_learning(Q_table, runs=20, timesteps=10000, gamma=1):
    for _ in range(runs):
        R_acc = np.zeros(fair_env.num_locs)
        fair_env.clean_all()
        state = fair_env.reset()
        
        for _ in range(timesteps):
            action = argmax_nsw(R_acc, gamma*Q_table[state], state)
            next, reward, done, info = fair_env.step(action)
            R_acc += reward
            
        fair_env.output_csv()
    return print("FINSIH EVALUATE NSW Q LEARNING")

def run_random(runs=20, timesteps=10000):
    for _ in range(runs):
        random.clean_all()
        for _ in range(timesteps):
            action = random.action_space.sample()
            next, reward, done, info = random.step(action)
            
        random.output_csv()
    return  print("FINISH RANDOM RUNS")

if __name__ == "__main__":
    
    non_fair_env = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                                    min_mean=10, sd=1, 
                                    center_mean=20, max_diff=2,
                                    output_path='Bandit/Classical_10_locations/Bandit_run_')

    fair_env = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                                min_mean=10, sd=1, 
                                center_mean=20, max_diff=2,
                                output_path='Bandit/NSW_10_locations/NSW_Bandit_run_')
    
    random = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                                min_mean=10, sd=1, 
                                center_mean=20, max_diff=2,
                                output_path='Bandit/Random_5_locations/Random_Bandit_run_')
    
    # Q_table = run_Q_learning(timesteps=100000, episodes=1, alpha=0.01, epsilon=0.4, gamma=1)    # need to have a high exploration rate
    # print('Q learning Q-table:\n{}'.format(Q_table))
    # # evaluate_Q_learning(Q_table, runs=1, timesteps=10000)
    
    # nsw_Q_table = run_NSW_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=1)
    # print('NSW Q learning Q-table:\n{}'.format(nsw_Q_table))
    # evaluate_NSW_Q_learning(nsw_Q_table, runs=1, timesteps=10000)

    Q_table = run_bandit_algorithm(timesteps=100000, epsilon=0.1)
    print('Bandit Algorithm Q table:\n{}'.format(Q_table))
    evaluate_bandit(Q_table, runs=50, timesteps=10000)
    
    nsw_Q_table = run_nsw_bandit_algorithm(timesteps=100000, epsilon=0.1)
    print('NSW Bandit Algorithm Q table: {}'.format(nsw_Q_table))
    evaluate_nsw_bandit(nsw_Q_table, runs=50, timesteps=10000)
    
    run_random(runs=50, timesteps=10000)