'''Evaluation of trained Q tables by playing the game, in environment with penalty'''
import numpy as np
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

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
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def eval_nsw(Q_table, taxi_loc=None, pass_dest=None, episodes=20, 
             nsw_lambda=0.01, gamma=1, check_dest=False, render=True):
    if check_dest == True:
        for i in range(env.size):
            for j in range(env.size):
                done = False
                R_acc = np.zeros(len(env.loc_coords))
                state = env.reset([i,j])
                if render == True: env.render()
                
                while not done:
                    action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
                    next, reward, done = env.step(action)
                    # reward = np.sum(reward) # for scalar reward with NSW Q-table
                    if render == True: env.render()
                    state = next
                    R_acc += reward
            
                print('Accumulated Reward, initial location {}: {}\n'.format([i,j], R_acc))
    else:
        for i in range(1, episodes+1):
            env._clean_metrics()
            done = False
            R_acc = np.zeros(len(env.loc_coords))
            pass_loc = None if pass_dest == None else 1
            state = env.reset(taxi_loc, pass_loc, pass_dest)
            print(state)
            if render == True: env.render()
            
            while not done:
                action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
                next, reward, done = env.step(action)
                # reward = np.sum(reward) # for scalar reward with NSW Q-table
                if render == True: env.render()
                state = next
                R_acc += reward
            
            print('Accumulated Reward, episode {}: {}\n'.format(i, R_acc))
        #env._output_csv()
    return print("FINSIH EVALUATE NSW Q LEARNING\n")

def eval_ql(Q_table, taxi_loc=None, pass_dest=None, episodes=20):
    for i in range(1, episodes+1):
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
    return print("FINISH EVALUATE Q LEARNING\n")

def check_all_locs(q_table, eval_steps, gamma, nsw, nsw_lambda=1e-4):
    invalid, prev, valid = [], 0, []
    print('Check initial locations...\n')
    for i in range(env.size):
        for j in range(env.size):
            count = 0
            R_acc = np.zeros(len(env.loc_coords))
            state = env.reset([i,j])
            prev = state
            for _ in range(eval_steps):
                if nsw == False:
                    action = np.argmax(gamma*q_table[state])
                else:
                    action = argmax_nsw(R_acc, gamma*q_table[state], nsw_lambda)
                next, reward, done = env.step(action)
                state = next
                if state == prev: count += 1
                else: count = 0
                if count == 10: 
                    invalid.append([i,j])    # initial location that doesn't work
                    break
                prev = state
                if nsw == True: R_acc += reward
            if count < 5: valid.append([i,j]) # append valid initial states
            
    if len(invalid) == 0: return print('Result: All initial locations WORK')
    elif len(invalid) == size*size: return print('Result: All initial locations FAIL')
    elif len(invalid) >= int(size*size/2): print('Result: These initial locations WORK: {}'.format(valid))
    else: return print('Result: These initial locations FAIL: {}'.format(invalid))

if __name__ == '__main__':
    # size = 5
    # loc_coords = [[0,0], [3,2]]
    # dest_coords = [[0,4], [3,3]]
    size = 8
    loc_coords = [[0,0], [0,5], [3,2], [3,7]]
    dest_coords = [[0,4], [5,0], [3,3], [7,0]]
    fuel = 10000
    
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, '', 8)
    env.seed(1122)  # make sure to use same seed as we used in learning
    
    q_table = np.load('Experiments/taxi_q_tables/NSW_Cont_Penalty_V2_size8_locs5_aN_2.npy')
    eval_nsw(q_table, taxi_loc=[5,5], pass_dest=None, nsw_lambda=1e-4,
               gamma=0.95, episodes=1, render=False,  check_dest=True)
    
    check_all_locs(q_table, eval_steps=10000, gamma=0.95, nsw_lambda=1e-4, nsw=True)
    #q_table = np.load('Experiments/taxi_q_tables/QL_size5_locs2.npy')
    #eval_ql(q_table, taxi_loc=[2,2], pass_dest=None, episodes=1)