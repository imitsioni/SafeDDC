import os

import matplotlib.pyplot as plt
import torch

from dynamics_library import PendulumCartContinuous, SampleAndHold, PendulumCartBroken
from dynamical_rrt import DynamicalRRT, TreeNode
from zonotope_lib import Zonotope, Box, plot_zonotope
import numpy as np
from pendulum_network_interface import save_network_to_folder
from pendulum_network_definitions import PendulumFC
from pendulum_train_utils import pendulum_train_explicit_1step
import sklearn.preprocessing as preprocessing
import torch.optim as opt
import pendulum_data_utils as du
import numpy as np


''' Generate data by arbitrary sampling of the dynamics'''

read_data_path = 'rrtc+random/converged_trials/'

is_output_relative = True

state_data, input_data, monster = du.pendulum_dataset(read_data_path)
data_len = len(state_data)
train_rate = 0.9
n_train = np.floor(data_len * train_rate)
train_data_in = np.zeros((0, 3))
train_data_out = np.zeros((0, 2))
valid_data_in = np.zeros((0, 3))
valid_data_out = np.zeros((0, 2))
for i in range(len(state_data)):
    new_data_in = np.concatenate((state_data[i][:-1, :3], input_data[i][:-1, :3]), axis=1)
    if is_output_relative:
        new_data_out = state_data[i][1:, :2] - state_data[i][:-1, :2]
    else:
        new_data_out = state_data[i][1:, :2]
    if i < n_train:
        train_data_in = np.concatenate((train_data_in, new_data_in), axis=0)
        train_data_out = np.concatenate((train_data_out, new_data_out), axis=0)
    else:
        valid_data_in = np.concatenate((valid_data_in, new_data_in), axis=0)
        valid_data_out = np.concatenate((valid_data_out, new_data_out), axis=0)


scaler_in = preprocessing.StandardScaler()
scaler_out = preprocessing.StandardScaler()

scaler_in.fit(train_data_in)
scaler_out.fit(train_data_out)

scaler = [scaler_in, scaler_out]

input_size = 3
block_size = 1
output_size = 2
network = PendulumFC(input_size, block_size, output_size)
lr = 5e-03  # starting learning rate
wd = 1e-3  # weight decay
gamma_1 = 0.75  # rate of learning rate adjustment for multisteplr scheduler (time horizon)
gamma_2 = 0.5  # rate of learning rate adjustment for plateau scheduler
optimizer = opt.Adam(network.parameters(), lr=lr, weight_decay=wd)
epochs = 500

seed = torch.seed()
print(seed)
network, loss_val = pendulum_train_explicit_1step(train_data_in, train_data_out, optimizer=optimizer, network=network, epochs=epochs, scaler=scaler)
plt.plot(loss_val)
plt.show()

save_network_to_folder(network, scaler=scaler, folder_path= read_data_path + 'relative_net/', is_output_relative=is_output_relative)

