'''NSW Q learning without R included in argmax, same initial values, with penalty environment v2'''
import numpy as np
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def run_NSW_Q_learning(episodes, alpha, epsilon, gamma, nsw_lambda, init_val, dim_factor, tolerance, file_name):
    Q_table = np.zeros([fair_env.observation_space.n, fair_env.action_space.n, len(fair_env.loc_coords)], dtype=float)
    Num = np.full(fair_env.observation_space.n, epsilon, dtype=float)
    Q_table = Q_table + init_val
    loss_data = []
    
    for i in range(1, episodes+1):
        R_acc = np.zeros(len(fair_env.loc_coords))
        state = fair_env.reset()
        done = False
        old_table = np.copy(Q_table)
        
        while not done:
            epsilon = Num[state]
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
                
            next_state, reward, done = fair_env.step(action)
            max_action = argmax_nsw(0, gamma*Q_table[next_state], nsw_lambda)
            new_value = Q_table[state, action] + alpha*(reward + gamma*Q_table[next_state, max_action] - Q_table[state, action])
            
            Num[state] *= dim_factor  # epsilon diminish over time
            Q_table[state, action] = new_value
            state = next_state
            R_acc += reward
        
            if fair_env.timesteps % 10000 == 0 and fair_env.timesteps != 0:
                loss = np.sum(np.abs(Q_table - old_table))
                loss_data.append(loss)
                print('Accumulated reward at timestep {}: {}\nLoss: {}\n'.format(fair_env.timesteps, R_acc, loss))
                if loss < tolerance:
                    loss_count += 1
                    if loss_count == 10: break # need to be smaller for consecutive loops to satisfy early break
                else: loss_count = 0
                old_table = np.copy(Q_table)
                
    np.save(file='taxi_q_tables/NSW_Cont_Penalty_V2_size{}_locs{}_{}'.format(fair_env.size,len(fair_env.loc_coords), file_name),
            arr=Q_table)
    np.save(file='taxi_q_tables/NSW_Cont_Penalty_V2_size{}_locs{}_{}_loss'.format(fair_env.size,len(fair_env.loc_coords), file_name),
            arr=loss_data)
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
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

if __name__ == '__main__':
    
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""NSW Q-learning on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=50000000, required=False, help="Timesteps each episode\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=1, required=False, help="Episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Discount rate\n")
    prs.add_argument("-nl", dest="nsw_lambda", type=float, default=1e-4, required=False, help="Smoothing factor\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    prs.add_argument("-d", dest="dim_factor", type=float, default=0.9, required=False, help="Diminish factor for epsilon\n")
    prs.add_argument("-t", dest="tolerance", type=float, default=1e-5, required=False, help="Loss threshold for Q-values between each episode\n")
    prs.add_argument("-s", dest="seed", type=int, default=1122, required=False, help="See for randomization of reset() function\n")
    prs.add_argument("-n", dest="file_name", type=str, default='', required=False, help="name of .npy\n")
    args = prs.parse_args()
    
    # size = 5
    # loc_coords = [[0,0], [3,2]]
    # dest_coords = [[0,4], [3,3]]
    
    # size = 6
    # loc_coords = [[0,0], [0,5], [3,2], [5,4]]
    # dest_coords = [[0,4], [5,0], [3,3], [2,0]]
    
    size = 8
    loc_coords = [[0,0], [0,5], [3,2], [5,4], [3,7], [5,6]]
    dest_coords = [[0,4], [5,0], [3,3], [2,0], [7,0], [6,6]]
    fuel = args.fuel
    
    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, 
                            output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)
    fair_env.seed(args.seed)
    
    run_NSW_Q_learning(episodes=args.episodes, alpha=args.alpha, epsilon=args.epsilon, 
                       gamma=args.gamma, nsw_lambda=args.nsw_lambda, init_val=args.init_val,
                       dim_factor=args.dim_factor, tolerance=args.tolerance, file_name=args.file_name)
    # nsw_Q_table = np.load('Taxi_MDP_Trained_Q-table/NSW_size5_locs2_without_reward_10.npy')
    # evaluate_NSW_Q_learning(nsw_Q_table, vec_dim=2, taxi_loc=[2,1], pass_dest=None, runs=1)
    
    
    


    