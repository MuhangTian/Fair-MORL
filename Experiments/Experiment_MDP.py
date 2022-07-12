''' Running experiments with NSW Q-learning and classical Q-learning'''
import numpy as np
from Fair_Taxi_MDP import Fair_Taxi_MDP

def run_Q_learning(episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99):
    Q_table = np.zeros([nonfair_env.observation_space.n, nonfair_env.action_space.n])
    count = 0
    for _ in range(episodes):
        state = nonfair_env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = nonfair_env.action_space.sample()
            else:
                if np.all(Q_table[state] == Q_table[state][0]) == True: 
                    # if all values same, choose randomly, since np.argmax returns 0 when values are same
                    action = nonfair_env.action_space.sample()
                else:
                    action = np.argmax(Q_table[state])
            
            next_state, reward, done, info = nonfair_env.step(action)
            reward = np.sum(reward)     # turn vector reward into scalar
            new_value = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[next_state]))
            
            Q_table[state, action] = new_value
            state = next_state
        count += 1
        print('Accumulated reward at episode {}: {}'.format(count,
                                                           nonfair_env._get_info()['Accumulated Reward']))
        
    np.save(file='Taxi_MDP_Trained_Q-Table/Qlearning_size{}_locs{}'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=Q_table)
    print('FINISH TRAINING Q LEARNING')
    return Q_table

def run_NSW_Q_learning(episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99, nsw_lambda=0.01, init_val=0):
    Q_table = np.zeros([fair_env.observation_space.n, fair_env.action_space.n, len(fair_env.loc_coords)])
    Q_table = Q_table + init_val
    count = 0
    R_acc = np.zeros(len(fair_env.loc_coords))
    
    for _ in range(episodes):
        state = fair_env.reset()
        done = False
        while not done:
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
                
            next_state, reward, done, info = fair_env.step(action)
            max_action = argmax_nsw(reward, gamma*Q_table[next_state], nsw_lambda)
            new_value = Q_table[state, action] + alpha*(reward + gamma*Q_table[next_state, max_action] - Q_table[state, action])
            
            Q_table[state, action] = new_value
            state = next_state
            R_acc += reward
            count += 1
            if count%1000 == 0:
                print('Accumulated reward at step {}: {}'.format(count, R_acc))
        
    np.save(file='Taxi_MDP_Trained_Q-Table/NSW_size{}_locs{}'.format(fair_env.size,len(fair_env.loc_coords)),
            arr=Q_table)
    print('FINISH TRAINING NSW Q LEARNING')
    return Q_table

def argmax_nsw(R, gamma_Q, nsw_lambda):
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[i], nsw_lambda) for i in range(fair_env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = fair_env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action

def argmax_nsw_geom(R, gamma_Q):    # unfinished argmax, calculate NSW by geometric mean
    sum = R + gamma_Q
    nsw_vals = [sum[i] for i in range(fair_env.action_space.n)]

def nsw(vec, nsw_lambda): 
    vec = vec + nsw_lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def evaluate_Q_learning(Q_table, taxi_loc=None, runs=20):
    for i in range(runs):
        nonfair_env._clean_metrics() # clean values before generating results for each run
        done = False
        state = nonfair_env.reset(taxi_loc)
        nonfair_env.render()
        
        while not done:
            num = Q_table[state]
            action = np.argmax(Q_table[state])
            next, reward, done, info = nonfair_env.step(action)
            nonfair_env.render()
            state = next
        # nonfair_env._output_csv()
    return print("FINISH EVALUATE Q LEARNING")

def evaluate_NSW_Q_learning(Q_table, vec_dim, taxi_loc=None, runs=20, nsw_lambda=0.01, gamma=1):
    for _ in range(runs):
        fair_env._clean_metrics()
        done = False
        R_acc = np.zeros(vec_dim)
        state = fair_env.reset(taxi_loc)
        fair_env.render()
        
        while not done:
            action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
            next, reward, done, info = fair_env.step(action)
            # reward = np.sum(reward) # for scalar reward with NSW Q-table
            fair_env.render()
            state = next
            R_acc += reward
        #fair_env._output_csv()
    return print("FINSIH EVALUATE NSW Q LEARNING")

def create_envs(size, loc_coords, dest_coords, fuel, fps=4):
    nonfair_env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, 
                                 output_path='Taxi_MDP/Q_learning/run_', fps=fps)

    fair_env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, 
                            output_path='Taxi_MDP/NSW_Q_learning/run_', fps=fps)
    
    random_env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, 
                               output_path='Taxi_MDP/Random/run_', fps=fps)
    
    return nonfair_env, fair_env, random_env

if __name__ == '__main__':
    size = 5
    loc_coords = [[0,0], [3,2]]
    dest_coords = [[0,4], [3,3]]
    fuel = 10000000
    nonfair_env, fair_env, random_env = create_envs(size, loc_coords, dest_coords, fuel, fps=8)
    
    # run_Q_learning(episodes=1000, alpha=0.1, epsilon=0.3, gamma=0.99)
    # Q_table = np.load('Taxi_MDP_Trained_Q-table/Qlearning_size5_locs2.npy')
    # evaluate_Q_learning(Q_table, taxi_loc=[0,0], runs=5)
    
    # run_NSW_Q_learning(episodes=1, alpha=0.1, epsilon=0.3, gamma=0.99, nsw_lambda=0.01, init_val=30)
    nsw_Q_table = np.load('Taxi_MDP_Trained_Q-table/NSW_size5_locs2.npy')
    evaluate_NSW_Q_learning(nsw_Q_table, vec_dim=2, taxi_loc=[4,0], runs=5)
    # evaluate_Q_learning(nsw_Q_table, taxi_loc=[2,4], runs=5)
    


    