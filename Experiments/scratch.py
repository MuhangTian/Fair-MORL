from Fair_Taxi_Bandit import Fair_Taxi_Bandit
env = Fair_Taxi_Bandit()    # Use default values
for i in range(1000):   # Some random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.output_csv()