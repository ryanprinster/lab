import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


a = np.load('/Users/ryanprinster/test/pos_data.npy')
print(len(a))
x = a[:,0]
y = a[:,1]

plt.plot(x, y)
plt.show()


# a = np.load('/Users/ryanprinster/test/vel_trans_data_2.npy')
# print(a)
# vel = a[:,0]
# x = range(len(a))

# plt.plot(x, vel)
# plt.show()




# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')

# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,

# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=True)