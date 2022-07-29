'''For visualization of results from taxi environments'''
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import ast

def plot_loss(loss_data, ID, x_range=None, linewidth=0.3, alpha=0.9, colors=['r'], labels=None, subplot=False):
    if len(loss_data)==1:
        if x_range == None:
            interval = np.arange(1, len(loss_data[0])+1)
        else:
            interval = np.arange(1, x_range+1)
            loss_data = loss_data[0][:x_range]
        plt.plot(interval, loss_data, colors[0], linewidth=linewidth, alpha=alpha)
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel('Loss')
        # plt.title('Q value loss over time for job {}'.format(ID))
        plt.title('Q value loss over time for 4 pickup locations')
        plt.show()
    elif subplot == True:
        if x_range == None:
            interval = np.arange(1, len(loss_data)+1)
        else:
            interval = np.arange(1, x_range+1)
            for i in range(len(loss_data)): loss_data[i] = loss_data[i][:x_range]
        fig, axs = plt.subplots(len(loss_data))
        for i in range(len(loss_data)):
            axs[i].plot(interval, loss_data[i], colors[i], linewidth=linewidth, alpha=alpha)
            axs[i].set_title(labels[i])
        #axs.set(xlabel='Intervals (each interval is 10000 steps)', ylabel='Q value loss')
        # plt.title('Q value loss across different experiments')
        fig.suptitle('Q value loss over time for 3 pickup locations during learning', fontweight='bold')
        fig.tight_layout()
        plt.show()
    else:
        if x_range == None:
            interval = np.arange(1, len(loss_data)+1)
        else:
            interval = np.arange(1, x_range+1)
            for i in range(len(loss_data)): loss_data[i] = loss_data[i][:x_range]
        for i in range(len(loss_data)):
            plt.plot(interval, loss_data[i], colors[i], linewidth=linewidth, alpha=alpha, label=labels[i])
            alpha -= 0.1
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel('Loss in Q values')
        # plt.title('Q value loss across different experiments')
        plt.title('Q value loss over time for 3 pickup locations during learning', fontweight='bold')
        plt.legend()
        plt.show()

def plot_total(IDs, x_range, linewidth, alpha, colors, labels, num_locs=3):
    data = []
    for id in IDs: 
        arr = find_data(id, num_locs)
        data.append(arr)
    if x_range == None:
        interval = np.arange(1, len(data[0])+1)
    else:
        interval = np.arange(1, x_range+1)
        for i in range(len(data)): data[i] = data[i][:x_range]
    fig, axs = plt.subplots(len(data))
    for i in range(len(data)):
        axs[i].plot(interval, data[i], colors[i], linewidth=linewidth, alpha=alpha)
        axs[i].set_title(labels[i])
    fig.suptitle('Total Accumulated Reward Over Time', fontweight='bold')
    fig.tight_layout()
    plt.show()

def find_data(ID, num_locs):
    with open('Experiments/log/{}.out'.format(ID)) as f:
        text = f.readlines()
    text = ''.join(text)
    matches = re.findall('\[.*\]', text)
    for i in range(len(matches)): 
        matches[i] = matches[i].strip('][')
        matches[i] = np.fromstring(matches[i], sep=' ')
        matches[i] = np.sum(matches[i])
    return matches

def plot_r_acc(data, labels, width, ):
    locs = []
    for i in range(len(data)): data[i] = np.mean(data[i], axis=0)
    for i in range(len(data[0])):
        arr = []
        for j in range(len(data)):
            arr.append(data[j][i])
        locs.append(arr)
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    for i in range(len(data[0])):
        ax.bar(x - width*(1-i), locs[i], width, label='Location {}'.format(i))
    
    ax.set_ylabel('Average Accumulated Reward')
    ax.set_title('Average Accumulated Reward Between Algorithms', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    return plt.show()

def nsw(vec): 
    return np.sum(np.log(vec))    # numpy uses natural log

def plot_nsw(data, labels, width, nsw_lambda=1e-4):
    nsws = []
    for i in range(len(data)):
        nsw = []
        for j in range(50):
            num = np.product(data[i][j])
            nsw.append(np.power(num, 1/len(data[i][j])))
        nsws.append(np.mean(nsw))
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    ax.bar(x, nsws, width)
    ax.set_ylabel('Average NSW Score')
    ax.set_title('Average NSW Score for Different Algorithms', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    fig.tight_layout()
    return plt.show()


if __name__ == '__main__':
    '''Plot Q value loss over time (learning)'''
    data1 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size6_locs3_2_loss.npy')
    data2 = np.load('Experiments/taxi_q_tables/QL_Penalty_size6_locs3_loss.npy')
    # data3 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size12_locs3_1_loss.npy')
    loss_data = [data1, data2]
    labels = ['Modified Q learning with NSW', 'Standatd Q learning']
    plot_loss(loss_data=loss_data, ID=1932259, colors=['r','b','g'], x_range=600, 
              linewidth=0.7, labels=labels, alpha=0.8, subplot=False)
    
    '''Plot Total Reward over Time (learning)'''
    # id = [1940069, 1940073,1940077]
    # plot_total(IDs=id, x_range=10000, linewidth=0.8, alpha=0.8,
    #            colors=['r','b','g'], labels=['8X8 Grid','10X10 Grid','12X12 Grid'])
    
    '''Plot accumulated reward'''
    # data1 = np.load('Experiments/ql_Racc_6.npy')
    # data2 = np.load('Experiments/nsw_6.npy')
    # data3 = np.load('Experiments/stationary_nsw_6.npy')
    # data = [data1, data2, data3]
    # labels = ['Standard Q learning', 'NSW non-stationary policy', 'NSW stationary policy']
    #plot_r_acc(data, labels=labels, width=0.2)
    
    '''Plot NSW'''
    # data1 = np.load('Experiments/ql_Racc_6.npy')
    # data2 = np.load('Experiments/nsw_6.npy')
    # data3 = np.load('Experiments/stationary_nsw_6.npy')
    # data = [data1, data2, data3]
    # labels = ['Standard Q learning', 'NSW non-stationary policy', 'NSW stationary policy']
    # plot_nsw(data=data, labels=labels, width=0.2)