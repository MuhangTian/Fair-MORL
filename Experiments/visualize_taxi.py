'''For visualization of results from taxi environments'''
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_data, ID):
    interval = np.arange(1, len(loss_data)+1)
    
    plt.plot(interval, loss_data, 'red', linewidth=0.5)
    plt.xlabel('Intervals (each interval is 10000 steps)')
    plt.ylabel('Loss')
    plt.title('Q value loss over time for job {}'.format(ID))
    plt.show()

if __name__ == '__main__':
    loss_data = np.load('Experiments/taxi_q_tables/NSW_Penalty_size5_locs2_1_loss.npy')
    plot_loss(loss_data, 1922343)