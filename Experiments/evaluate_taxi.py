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

def eval_nsw(Q, name, taxi_loc=None, pass_dest=None, episodes=20, mode='non-stationary',
             nsw_lambda=0.01, check_dest=False, render=True, gamma=0.95, update=False):
    Racc = []
    if render == True: check_dest = False
    if check_dest == True:
        for i in range(env.size):
            for j in range(env.size):
                done = False
                R_acc = np.zeros(len(env.loc_coords))
                state = env.reset([i,j])
                c = 0
                print('Initial State: {}'.format(env.decode(state)))
                
                while not done:   
                    if mode == 'non-stationary':               
                        action = argmax_nsw(R_acc, Q[state], nsw_lambda)
                    elif mode == 'stationary':
                        action = argmax_nsw(0, Q[state], nsw_lambda)
                    next, reward, done = env.step(action)
                    if update == True:
                        max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
                        Q[state, action] = Q[state, action] + 0.1*(reward + gamma*Q[next, max_action] - Q[state, action])
                    state = next
                    R_acc += np.power(gamma, c)*reward
                    c += 1

                nsw_score = nsw(R_acc, nsw_lambda)
                Racc.append(R_acc)
                print('Accumulated Reward: {}\nNSW: {}\n'.format(R_acc, nsw_score))
    else:
        for i in range(1, episodes+1):
            env._clean_metrics()
            done = False
            R_acc = np.zeros(len(env.loc_coords))
            pass_loc = None if pass_dest == None else 1
            state = env.reset(taxi_loc, pass_loc, pass_dest)
            c = 0
            if render == True: env.render()
            
            while not done:
                if mode == 'non-stationary':               
                    action = argmax_nsw(R_acc, Q[state], nsw_lambda)
                elif mode == 'stationary':
                    action = argmax_nsw(0, Q[state], nsw_lambda)
                next, reward, done = env.step(action)
                if update == True:
                    max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
                    Q[state, action] = Q[state, action] + 0.1*(reward + gamma*Q[next, max_action] - Q[state, action])
                if render == True: env.render()
                state = next
                #R_acc += reward
                R_acc += np.power(gamma, c)*reward
                c += 1
            
            print('Accumulated Reward, episode {}: {}\n'.format(i, R_acc))
            Racc.append(R_acc)
        #env._output_csv()
        np.save('Experiments/{}.npy'.format(name), Racc)
    
    print("Average Accumulated Reward: {}\nFINSIH EVALUATE NSW Q LEARNING\n".format(np.mean(Racc, axis=0)))

def eval_ql(Q, taxi_loc=None, pass_dest=None, episodes=20, render=False):
    Racc = []
    for i in range(1, episodes+1):
        env._clean_metrics() # clean values before generating results for each run
        done = False
        R_acc = np.zeros(len(env.loc_coords))
        pass_loc = None if pass_dest == None else 1
        state = env.reset(taxi_loc, pass_loc, pass_dest)
        if render==True: env.render()
        while not done:
            action = np.argmax(Q[state])
            next, reward, done = env.step(action)
            if render==True: env.render()
            state = next
            R_acc += reward
        
        Racc.append(R_acc)
        print('Trajectory {}: {}'.format(i, R_acc))
    # env._output_csv()
    np.save('Experiments/ql_Racc_6.npy', Racc)
    return print("FINISH EVALUATE Q LEARNING\n")

def check_all_locs(Q, eval_steps, nsw, nsw_lambda=1e-4, thres=20, gamma=0.95, update=False, mode='non-stationary'):
    invalid, prev, valid = [], 0, []
    print('Check initial locations...\n')
    for i in range(env.size):
        for j in range(env.size):
            count = 0
            R_acc = np.zeros(len(env.loc_coords))
            c = 0
            state = env.reset([i,j])
            prev = state
            for _ in range(eval_steps):
                if nsw == False:
                    action = np.argmax(Q[state])
                else:
                    if mode == 'non-stationary':
                        action = argmax_nsw(R_acc, Q[state], nsw_lambda)
                    elif mode == 'stationary':
                        action = argmax_nsw(0, Q[state], nsw_lambda)
                next, reward, done = env.step(action)
                if update == True:
                    max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
                    Q[state, action] = Q[state, action] + 0.1*(reward + gamma*Q[next, max_action] - Q[state, action])
                state = next
                if state == prev: count += 1
                else: count = 0
                if count == thres: 
                    invalid.append([i,j])    # initial location that doesn't work
                    break
                prev = state
                if nsw == True: 
                    R_acc += np.power(gamma, c)*reward
                    c += 1
            if count < thres: valid.append([i,j]) # append valid initial states
            
    if len(invalid) == 0: return print('Result: All initial locations WORK\n')
    elif len(invalid) == size*size: return print('Result: All initial locations FAIL\n')
    elif len(invalid) >= int(size*size/2): print('Result: These initial locations WORK: {}\n'.format(valid))
    else: return print('Result: These initial locations FAIL: {}\n'.format(invalid))

def get_setting(size, num_locs): # stores fixed environment settings (see excel sheet)
    if num_locs == 2:
        loc_coords = [[0,0],[3,2]]
        dest_coords = [[0,4],[3,3]]
    elif num_locs == 3:
        loc_coords = [[0,0],[0,5],[3,2]]
        dest_coords = [[0,4],[5,0],[3,3]]
    elif num_locs == 4:
        loc_coords = [[0,0], [0,5], [3,2], [9,0]]
        dest_coords = [[0,4], [5,0], [3,3], [0,9]]
    elif num_locs == 5:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[4,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[8,9]]
    else:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[8,9],[6,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[4,7],[8,3]]
    return size, loc_coords, dest_coords

if __name__ == '__main__':
    size, loc_coords, dest_coords = get_setting(5,2)
    fuel = 10000
    env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, '', 15)
    env.seed(1122)  # make sure to use same seed as we used in learning
    update = True
    Q = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size5_locs2_1.npy')

    eval_nsw(Q, taxi_loc=[4,0], nsw_lambda=1e-4, mode='non-stationary',
            episodes=50, render=True,  check_dest=True, name='', update=update)
    check_all_locs(Q, eval_steps=4000, nsw_lambda=1e-4, nsw=True, thres=50, update=update, mode='non-stationary')
    #Q = np.load('Experiments/taxi_Qs/QL_size5_locs2.npy')
    # eval_ql(Q, taxi_loc=[0,0], pass_dest=None, episodes=50, render=True)
    