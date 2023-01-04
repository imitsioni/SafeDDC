import matplotlib.pyplot as plt
import pickle
import numpy as np

dist_folders = ['fullrand', 'localrand_0', 'localrand_pi', 'rrtc', 'rrtc+rand']

for df in dist_folders:
    with open(df + '/data/data_in.pickle', 'rb') as handle:
        data_in = pickle.load(handle)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(data_in[:, 0], data_in[:, 1], data_in[:, 2], 'r*')
    ax.set_xlabel('angle')
    ax.set_ylabel('velocity')
    ax.set_zlabel('input')

    ax.set_xlim((-2*np.pi, 2*np.pi))
    ax.set_ylim((-10, 10))
    ax.set_zlim((-10, 10))
    ax.set_title(df)
plt.show()