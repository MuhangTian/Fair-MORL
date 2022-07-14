import numpy as np
import argparse
from Fair_Taxi_MDP import Fair_Taxi_MDP

def run_Q_learning(episodes=20, alpha=0.1, epsilon=0.1, gamma=0.99, init_val=0):
    Q_table = np.zeros([nonfair_env.observation_space.n, nonfair_env.action_space.n])
    Q_table = Q_table + init_val
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

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-learning on Taxi""")
    prs.add_argument("-f", dest="fuel", type=int, default=10000, required=False, help="Timesteps each episode\n")
    prs.add_argument("-ep", dest="episodes", type=int, default=10000, required=False, help="Episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.3, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Discount rate\n")
    prs.add_argument("-i", dest="init_val", type=int, default=0, required=False, help="Initial values\n")
    args = prs.parse_args()
    
    size = 5
    loc_coords = [[0,0], [3,2]]
    dest_coords = [[0,4], [3,3]]
    fuel = args.fuel
    
    nonfair_env = Fair_Taxi_MDP(size, loc_coords, dest_coords, fuel, 
                                 output_path='Taxi_MDP/Q_learning/run_', fps=4)
    
    run_Q_learning(episodes=args.episodes, alpha=args.alpha, 
                   epsilon=args.epsilon, gamma=args.gamma, init_val=args.init_val)
    
    # Q_table = np.load('Taxi_MDP_Trained_Q-table/Qlearning_size5_locs2.npy')
    
    # evaluate_Q_learning(Q_table, taxi_loc=[4,4], runs=5)