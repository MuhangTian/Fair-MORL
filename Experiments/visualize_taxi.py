'''For visualization of results from taxi environments'''
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_data, ID, x_range=None, linewidth=0.3, alpha=0.9, color='r', labels=None):
    if np.ndim(loss_data)==1:
        if x_range == None:
            interval = np.arange(1, len(loss_data)+1)
        else:
            interval = np.arange(1, x_range+1)
            loss_data = loss_data[:x_range]
        plt.plot(interval, loss_data, color, linewidth=linewidth, alpha=alpha)
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel('Loss')
        plt.title('Q value loss over time for job {}'.format(ID))
        plt.show()
    else:
        colors = ['r', 'b', 'g', 'y', 'p', 'c']
        if x_range == None:
            interval = np.arange(1, len(loss_data)+1)
        else:
            interval = np.arange(1, x_range+1)
            for i in range(len(loss_data)): loss_data[i] = loss_data[i][:x_range]
        for i in range(len(loss_data)):
            plt.plot(interval, loss_data[i], colors[i], linewidth=linewidth, alpha=alpha, label=labels[i])
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel('Loss')
        plt.title('Q value loss across different experiments')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # data1 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size10_locs4_1_loss.npy')
    # data2 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size10_locs5_1_loss.npy')
    # data3 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size10_locs6_1_loss.npy')
    data4 = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size10_locs6_1_loss.npy')
    # loss_data = [data1, data2, data3, data4]
    labels = ['Size 10, 4 locations', 'Size 10, 5 locations', 'Size 10, 6 locations', 'Size 10, 3 locations']
    plot_loss(loss_data=data4, ID=1932259, color='red', x_range=20000, 
              linewidth=0.5, labels=labels, alpha=0.7)