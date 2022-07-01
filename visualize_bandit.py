import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def line_graph(path1, path2, path3, runs, loc_nums, steps=10000): # plot graph based on timesteps and cumulative average of total reward
    avg1 = cumulative_average(path1, runs, loc_nums, steps)
    avg2 = cumulative_average(path2, runs, loc_nums, steps)
    avg3 = cumulative_average(path3, runs, loc_nums, steps)
    time = np.arange(1, 10001)
    
    plt.plot(time, avg1, 'red', alpha=1, linewidth=1, label='Bandit Algorithm')
    plt.plot(time, avg2, 'green', alpha=1, linewidth=1, label='NSW Bandit Algorithm')
    plt.plot(time, avg3, 'blue', alpha=1, linewidth=1, label='Random')
    
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative average accumulated total reward')
    plt.title('Bandit Algorithm vs NSW Bandit Algorithm (5 locations)', fontweight='bold')
    plt.legend()
    return plt.show()

def cumulative_average(path, runs, loc_nums, steps=10000):
    avg = average_between_runs(path, runs, loc_nums)
    avg = np.sum(avg, axis=0)
    time = np.arange(1, 10001)
    div_avg = np.divide(avg, time)
    return div_avg

def bar_chart(path1, path2, path3, runs, loc_nums):  # plot bar chart comparing averages of final accumulated reward at each location
    arr1 = average_between_runs(path1, runs, loc_nums)
    arr2 = average_between_runs(path2, runs, loc_nums)
    arr3 = average_between_runs(path3, runs, loc_nums)
    locs = [[arr1[i][9999], arr2[i][9999], arr3[i][9999]] for i in range(loc_nums)]
    width = 0.1
    labels = ['Bandit Algorithm', 'NSW Bandit Algorithm', 'Random']
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x - width*2, locs[0], width, label='Location 1')
    rects2 = ax.bar(x - width, locs[1], width, label='Location 2')
    rects3 = ax.bar(x + width*0, locs[2], width, label='Location 3')
    rects4 = ax.bar(x + width, locs[3], width, label='Location 4')
    rects5 = ax.bar(x + width*2, locs[4], width, label='Location 5')
    
    ax.set_ylabel('Average Accumulated Total Reward')
    ax.set_title('Bandit Algorithm vs NSW Bandit Algorithm (5 locations)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    return plt.show()

def average_between_runs(path, runs, loc_nums):   # take averages of values in each run
    '''
    Returns average metrics for each location, stored in 2D array
    '''
    arr, result = [], []
    for i in range(1, runs+1):
        df = pd.read_csv(path+str(i)+'.csv')
        for j in range(loc_nums):
            data = df['Location {}'.format(j)].to_numpy()
            try:
                arr[j].append(data)
            except:
                arr.append([data])
    
    for arrays in arr:
        mean = np.mean(arrays, axis=0)
        result.append(mean)
            
    return result

if __name__ == "__main__":
    
    bar_chart(path1='Bandit/Classical_5_locations/Bandit_run_', 
              path2='Bandit/NSW_5_locations/NSW_Bandit_run_', 
              path3='Bandit/Random_5_locations/Random_Bandit_run_', runs=50, loc_nums=5)
    
    line_graph(path1='Bandit/Classical_5_locations/Bandit_run_',
               path2='Bandit/NSW_5_locations/NSW_Bandit_run_', 
               path3='Bandit/Random_5_locations/Random_Bandit_run_', runs=50, loc_nums=5)