'''
Running tests for bandit taxi problem, using our algorithm and Q-learning
'''
import numpy as np
from Fair_Taxi_Bandit import Fair_Taxi_Bandit

def run_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=1):
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    
    for _ in range(episodes):
        state = env.reset()
        for _ in range(timesteps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])  # since only one state
            
            next_state, reward, done, info = env.step(action)
            reward = np.sum(reward)     # turn vector reward into scalar
            new_value = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[next_state]))
            
            Q_table[state, action] = new_value
            state = next_state
    
    print('FINISH TRAINING Q-LEARNING')
    return Q_table

def run_NSW_Q_learning(timesteps=10000, episodes=20, alpha=0.1, epsilon=0.1, gamma=1):
    Q_table = np.ones([env2.observation_space.n, env2.action_space.n, env2.num_locs]) # Cannot use zero due to log
    R_acc = np.zeros(env2.num_locs)
    
    for _ in range(episodes):
        state = env2.reset()
        for _ in range(timesteps):
            if np.random.uniform(0,1) < epsilon:
                action = env2.action_space.sample()
            else:
                action = argmax_nsw(R_acc, gamma*Q_table[state], state)
                
            next_state, reward, done, info = env2.step(action)
            max_action = argmax_nsw(reward, gamma*Q_table[next_state], next_state)
            new_value = Q_table[state, action] + alpha*(reward + gamma*Q_table[next_state, max_action] - Q_table[state, action])
            
            Q_table[state, action] = new_value
            state = next_state
            R_acc += reward
    
    print('FINISH TRAINING NSW Q-LEARNING')
    return Q_table

def nsw(vec): return np.sum(np.log(vec))

def argmax_nsw(R, gamma_Q, state):
    '''
    Function that finds an action that max NSW(R_acc + gamma*Q_table[state, action])
    :param R: (array) any vector reward to be added
    :param gamma_Q: (2D array) gamma*Q_table[state]
    :param state: (int) current state
    '''
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[state, i]) for i in range(env.action_num)]
    action = np.argmax(nsw_vals)
    return action

def evaluate():
    return

def visualize():
    return

if __name__ == "__main__":
    env = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                       min_mean=10, sd=0.1, 
                       center_mean=20, max_diff=2,
                       output_direct='Bandit_Qlearning/run_')

    env2 = Fair_Taxi_Bandit(num_locs=5, max_mean=40, 
                        min_mean=10, sd=0.1, 
                        center_mean=20, max_diff=2,
                        output_direct='Bandit_NSW/run_')
    
    x = run_Q_learning()
    # y = run_NSW_Q_learning()
    print('Q learning Q-table:\n{}'.format(x))
    # print('NSW Q learning Q-table:\n{}'.format(y))
    # env.output_csv()
    # env2.output_csv()