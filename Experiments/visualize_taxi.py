'''For visualization of results from taxi environments'''
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_data, ID, x_range=None, linewidth=0.3, alpha=0.9, color='r'):
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

if __name__ == '__main__':
    loss_data = np.load('Experiments/taxi_q_tables/NSW_Penalty_V2_size8_locs4_1_loss.npy')
    plot_loss(loss_data, 1932259, color='c', x_range=5000)