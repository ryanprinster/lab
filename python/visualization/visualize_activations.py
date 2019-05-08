import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
histograms = np.load('/Users/ryanprinster/test/histograms.npy')

print(histograms.shape)

def plot_positional_activations(histograms):
    w=20
    h=20
    fig=plt.figure(figsize=(11, 11))
    columns = 32
    rows = 16
    for i in range(1, columns*rows +1):
        img = histograms[i-1,:,:]
        ax = fig.add_subplot(rows, columns, i)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.axis('off')
        plt.imshow(img, origin="lower", cmap=plt.get_cmap('jet'))

    plt.show()

def plot_directional_activations():  
    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r

    fig=plt.figure(figsize=(10, 10))
    columns = 2
    rows = 3
    for i in range(1, columns*rows +1):
        ax = fig.add_subplot(rows, columns, i)
        ax = plt.polar(theta, r)
        plt.show()

        # ax = plt.subplot(111, projection='polar')
        # ax.plot(theta, r)
        # ax.set_rmax(2)
        # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
        # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        # ax.grid(True)

    plt.show()

plot_positional_activations(histograms)

