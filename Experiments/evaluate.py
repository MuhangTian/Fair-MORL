'''Evaluation of trained Q tables by playing the game'''
import numpy as np
from Fair_Taxi_MDP import Fair_Taxi_MDP

def argmax_nsw(R, gamma_Q, nsw_lambda):
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[i], nsw_lambda) for i in range(env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action

def argmax_nsw_geom(R, gamma_Q):    # unfinished argmax, calculate NSW by geometric mean
    sum = R + gamma_Q
    nsw_vals = [sum[i] for i in range(env.action_space.n)]

def nsw(vec, nsw_lambda): 
    vec = vec + nsw_lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def evaluate_NSW_Q_learning(Q_table, taxi_loc=None, pass_dest=None, runs=20, nsw_lambda=0.01, gamma=1):
    for _ in range(runs):
        env._clean_metrics()
        done = False
        R_acc = np.zeros(len(env.loc_coords))
        pass_loc = None if pass_dest == None else 1
        state = env.reset(taxi_loc, pass_loc, pass_dest)
        env.render()
        
        while not done:
            action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
            next, reward, done = env.step(action)
            # reward = np.sum(reward) # for scalar reward with NSW Q-table
            env.render()
            state = next
            R_acc += reward
        #env._output_csv()
    return print("FINSIH EVALUATE NSW Q LEARNING")

def evaluate_Q_learning(Q_table, taxi_loc=None, pass_dest=None, runs=20):
    for i in range(runs):
        env._clean_metrics() # clean values before generating results for each run
        done = False
        pass_loc = None if pass_dest == None else 1
        state = env.reset(taxi_loc, pass_loc, pass_dest)
        env.render()
        
        while not done:
            action = np.argmax(Q_table[state])
            next, reward, done = env.step(action)
            env.render()
            state = next
        # env._output_csv()
    return print("FINISH EVALUATE Q LEARNING")

def check_all_locs(q_table, size, eval_steps, gamma, nsw, nsw_lambda=1e-4):
    invalid, prev, valid = [], 0, []
    for i in range(size):
        for j in range(size):
            count = 0
            R_acc = np.zeros(len(env.loc_coords))
            state = env.reset([i,j])
            prev = state
            for _ in range(eval_steps):
                num = q_table[state]
                if nsw == False:
                    action = np.argmax(gamma*q_table[state])
                else:
                    action = argmax_nsw(R_acc, gamma*q_table[state], nsw_lambda)
                next, reward, done = env.step(action)
                state = next
                if state == prev: count += 1
                else: count = 0
                if count == 5: 
                    invalid.append([i,j])    # initial location that doesn't work
                    break
                prev = state
                if nsw == True: R_acc += reward
            if count < 5: valid.append([i,j]) # append valid initial states
            
    if len(invalid) == 0: return print('All initial locations WORK')
    elif len(invalid) == size*size: return print('All initial locations FAILED')
    elif len(invalid) >= int(size*size/2): print('These initial locations WORK: {}'.format(valid))
    else: return print('These initial locations FAIL: {}'.format(invalid))

if __name__ == '__main__':
    size = 5
    loc_coords = [[0,0], [3,2]]
    dest_coords = [[0,4], [3,3]]
    fuel = 10000
    env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, '', 10)
    q_table = np.load('Experiments/taxi_q_tables/QL_size5_locs2.npy')
    #evaluate_NSW_Q_learning(q_table, taxi_loc=[1,0], pass_dest=None, nsw_lambda=1e-4, gamma=0.95, runs=1)
    #q_table = np.load('Experiments/taxi_q_tables/QL_size5_locs2.npy')
    #evaluate_Q_learning(q_table, taxi_loc=[1,3], pass_dest=None, runs=1)
    check_all_locs(q_table, size, eval_steps=10000, gamma=0.95, nsw=False)