import gym
import numpy as np
import random

env = gym.make("Taxi-v3")
Q_table = np.zeros([env.observation_space.n, env.action_space.n])

def train_agent(alpha=0.1, epsilon=0.2, discount=0.4, time=100000):
    
    for i in range(time+1):
        # Initialize variables for each episode
        state = env.reset()
        reward = 0
        done = False
        
        while not done:
            
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])
        
            next_state, reward, done, info = env.step(action)
            
            old_action_value = Q_table[state, action]
            new_action_value = old_action_value + alpha*(reward+discount*np.max(Q_table[next_state]) - old_action_value)
            
            Q_table[state, action] = new_action_value
            state = next_state

    print("Finished Training \n")

def run_agent_reward(max_step=1000, total_reward=0):
    state = env.reset()
    for i in range(max_step):
    
        action = np.argmax(Q_table[state])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
        
        if done == True:
            break

    env.close()

def reward_random_episodes(num_episodes=100):
    total_reward=np.array([])
    for i in range(num_episodes):
        np.append(total_reward, run_agent_reward())
    return total_reward
    
train_agent()
run_agent_reward()