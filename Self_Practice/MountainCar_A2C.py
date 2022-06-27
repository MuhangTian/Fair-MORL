from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_agent(total_steps=1e4):
    total_steps = int(total_steps)
    # Hyperparameters from the internet
    model = A2C("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log='TensorBoard_MountainCar/',
            )
    
    model.learn(total_timesteps=total_steps, callback=eval_callback)
    # model.save('MountainCar')
    # del model
    return print('Finished Training for {} steps'.format(total_steps))

def run_agent(time=1000):
    # model = DQN.load('MountainCar')
    model = A2C.load('A2C_MountainCar/best_model')
    obs = env.reset()
    
    for i in range(time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # if dones == True:
        #     break
        env.render()

def csv_results():
    data = np.load('A2C_MountainCar/evaluations.npz')
    # Get mean reward across episodes
    mean_reward = [np.mean(rewards) for rewards in data['results']]
    # avg_ep_length = [np.mean(data) for data in data['ep_lengths']]
    df = pd.DataFrame({'Time Steps': data['timesteps'], 'Average Total Reward': mean_reward})
    df.to_csv('A2C_MountainCar.csv')

def plot():
    df = pd.read_csv('A2C_MountainCar.csv')
    time = df['Time Steps'].to_numpy()
    mean = df['Average Total Reward'].to_numpy()
    plt.plot(time, mean)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Total Reward')

    return plt.show()

if __name__ == '__main__':
    eval_env = VecMonitor(make_vec_env('MountainCar-v0', n_envs=4))
    env = make_vec_env('MountainCar-v0', n_envs=4)
    # EvalCallback is used to evaluate performance of the agent
    eval_callback = EvalCallback(eval_env, best_model_save_path="./A2C_MountainCar/",
                                log_path="./A2C_MountainCar/", eval_freq=1000,
                                deterministic=True, render=False)
    
    train_agent(1e6)
    #run_agent(10000)
    #csv_results()
    #plot()