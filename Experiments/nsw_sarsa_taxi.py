import numpy as np
import argparse
from Fair_Taxi_MDP import Fair_Taxi_MDP

def run_NSW_SARSA(episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99, nsw_lambda=0.01, init_val=0, file_name=''):
    Q_table = np.zeros([fair_env.observation_space.n, fair_env.action_space.n, len(fair_env.loc_coords)])
    # Q_table = Q_table + init_val
    Q_table[:, :4, :] = init_val
    #count = 0
    
    for i in range(1, episodes+1):
        R_acc = np.zeros(len(fair_env.loc_coords))
        state = fair_env.reset()
        done = False
        if np.random.uniform(0,1) < epsilon:
            action = fair_env.action_space.sample()
        else:
            action = argmax_nsw(R_acc, gamma*Q_table[state], nsw_lambda)
            
        while not done:
            next_state, reward, done, info = fair_env.step(action)
            
            if np.random.uniform(0,1) < epsilon:
                next_action = fair_env.action_space.sample()
            else:
                next_action = argmax_nsw(R_acc, gamma*Q_table[next_state], nsw_lambda)
            
            new_value = Q_table[state, action] + alpha*(reward + gamma*Q_table[next_state, next_action] - Q_table[state, action])
            
            Q_table[state, action] = new_value
            state = next_state
            action = next_action
            R_acc += reward
        
        if i%1000: print('Accumulated reward episode {}: {}'.format(i, R_acc))
        
    np.save(file='taxi_q_tables/NSW_size{}_locs{}_SARSA{}'.format(fair_env.size,len(fair_env.loc_coords), file_name),
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

def evaluate_NSW_Q_learning(Q_table, vec_dim, taxi_loc=None, pass_dest=None, runs=20, nsw_lambda=0.01, gamma=1):
    for _ in range(runs):
        fair_env._clean_metrics()
        done = False
        R_acc = np.zeros(vec_dim)
        pass_loc = None if pass_dest == None else 1
        state = fair_env.reset(taxi_loc, pass_loc, pass_dest)
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

if __name__ == '__main__':
    
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""NSW SARSA on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each episode\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=10000, required=False, help="Episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.3, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Discount rate\n")
    prs.add_argument("-nl", dest="nsw_lambda", type=float, default=0.01, required=False, help="Smoothing factor\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    prs.add_argument("-n", dest="file_name", type=str, default='', required=False, help="name of .npy\n")
    args = prs.parse_args()
    
    size = 5
    loc_coords = [[0,0], [3,2]]
    dest_coords = [[0,4], [3,3]]
    fuel = args.fuel
    
    fair_env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, 
                            output_path='Taxi_MDP/NSW_SARSA/run_', fps=4)
    
    run_NSW_SARSA(episodes=args.episodes, alpha=args.alpha, epsilon=args.epsilon, 
                  gamma=args.gamma, nsw_lambda=args.nsw_lambda, init_val=args.init_val,
                  file_name=args.file_name)
    # nsw_sarsa_Q_table = np.load('Taxi_MDP_Trained_Q-table/NSW_size5_locs2_SARSA_10.npy')
    # evaluate_NSW_Q_learning(nsw_sarsa_Q_table, vec_dim=2, taxi_loc=None, pass_dest=None, runs=1)