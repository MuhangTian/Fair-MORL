'''
NOTE: EvalCallback class evaluates agent's performance with a set number of episodes and a set frequency
of steps. For every eval_freq, evaluates the agent with n_eval_episodes based on the current
state of agent

It output a bestmodel.zip (which is the agent), and evaluations.npz, which contains three data:
timesteps, results, and episode length. Results is a 2D array since there are multiple episodes
in the evaluation, each number is the total reward earned by the agent in that episode

evaluate_policy() function alone is not useful, it only gives a mean_reward and std_reward when running
the agent for a given number of episodes, it is also slow since it actually need to run the agent
'''

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_agent(total_steps=1e4):
    total_steps = int(total_steps)
    # Hyperparameters from the internet
    model = DQN("MlpPolicy",
            env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log='TensorBoard_MountainCar/',
            seed=2)
    
    model.learn(total_timesteps=total_steps, callback=eval_callback)
    # model.save('MountainCar')
    # del model
    return print('Finished Training for {} steps'.format(total_steps))

def run_agent(time=1000):
    # model = DQN.load('MountainCar')
    model = DQN.load('DQN_MountainCar/best_model')
    obs = env.reset()
    
    for i in range(time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones == True:
            break
        env.render()

def csv_results():
    data = np.load('DQN_MountainCar/evaluations.npz')
    # Get mean reward across episodes
    mean_reward = [np.mean(rewards) for rewards in data['results']]
    # avg_ep_length = [np.mean(data) for data in data['ep_lengths']]
    df = pd.DataFrame({'Time Steps': data['timesteps'], 'Average Total Reward': mean_reward})
    df.to_csv('DQN_MountainCar.csv')

def plot():
    df = pd.read_csv('DQN_MountainCar.csv')
    time = df['Time Steps'].to_numpy()
    mean = df['Average Total Reward'].to_numpy()
    plt.plot(time, mean)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Total Reward')

    return plt.show()

if __name__ == '__main__':
    eval_env = Monitor(gym.make('MountainCar-v0'))
    env = gym.make('MountainCar-v0')
    # EvalCallback is used to evaluate performance of the agent
    eval_callback = EvalCallback(eval_env, best_model_save_path="./DQN_MountainCar/",
                                log_path="./DQN_MountainCar/", eval_freq=1000,
                                deterministic=True, render=False)
    
    #train_agent(2e5)       # Has an overfitting problem
    #run_agent(10000)
    # csv_results()
    plot()
