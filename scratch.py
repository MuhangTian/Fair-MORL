from Experiments.Fair_Taxi_Bandit import Fair_Taxi_Bandit
env = Fair_Taxi_Bandit()
env.step(1)
print(env.rewards)
env.step(2)
print(env.rewards)
print(env.rewards)
env.step(1)
print(env.rewards)