import numpy as np
import argparse
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def run_Q_learning(trajectorys=20, alpha=0.1, epsilon=0.1, gamma=0.99, init_val=0, tolerance=1e-10):
    Q_table = np.zeros([nonfair_env.observation_space.n, nonfair_env.action_space.n])
    Q_table = Q_table + init_val
    loss_data = []
    
    for i in range(1, trajectorys+1):
        R_acc = np.zeros(len(nonfair_env.loc_coords))
        state = nonfair_env.reset()
        old_table = np.copy(Q_table)
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
            
            next_state, reward, done = nonfair_env.step(action)
            R_acc += reward
            reward = np.sum(reward)     # turn vector reward into scalar
            new_value = (1 - alpha)*Q_table[state, action] + alpha*(reward+gamma*np.max(Q_table[next_state]))
            
            Q_table[state, action] = new_value
            state = next_state

        loss = np.sum(np.abs(Q_table - old_table))
        loss_data.append(loss)
        print('Accumulated reward at trajectory {}: {}\nLoss: {}\n'.format(i, R_acc, loss))
        if loss < tolerance:
            loss_count += 1
            if loss_count == 10: break  # need to be smaller for consecutive loops to satisfy early break
        else: loss_count = 0
        
    np.save(file='Experiments/taxi_q_tables/QL_Penalty_size{}_locs{}'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=Q_table)
    np.save(file='Experiments/taxi_q_tables/QL_Penalty_size{}_locs{}_loss'.format(nonfair_env.size, len(nonfair_env.loc_coords)),
            arr=loss_data)
    print('FINISH TRAINING Q LEARNING')
    return Q_table

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-learning on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each trajectory\n")
    prs.add_argument("-ep", dest="trajectorys", type=int, default=2000, required=False, help="trajectorys.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Discount rate\n")
    prs.add_argument("-i", dest="init_val", type=int, default=30, required=False, help="Initial values\n")
    args = prs.parse_args()
    
    # size = 5
    # loc_coords = [[0,0], [3,2]]
    # dest_coords = [[0,4], [3,3]]
    
    size = 6
    loc_coords = [[0,0], [0,5], [3,2]]
    dest_coords = [[0,4], [5,0], [3,3]]
    fuel = args.fuel
    
    nonfair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, 
                                        output_path='Taxi_MDP/Q_learning/run_', fps=4)
    
    run_Q_learning(trajectorys=args.trajectorys, alpha=args.alpha, 
                   epsilon=args.epsilon, gamma=args.gamma, init_val=args.init_val)
