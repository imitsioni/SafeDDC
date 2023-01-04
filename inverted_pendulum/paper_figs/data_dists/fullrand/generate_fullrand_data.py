import os
from dynamics_library import PendulumCartContinuous, SampleAndHold, PendulumCartBroken
from dynamical_rrt import DynamicalRRT, TreeNode
from zonotope_lib import Zonotope, Box, plot_zonotope
import numpy as np
from pendulum_network_interface import save_network_to_folder
from pendulum_network_definitions import PendulumFC
from pendulum_train_utils import pendulum_train_explicit_1step
import sklearn.preprocessing as preprocessing
import torch.optim as opt
import pickle

'''Dynamics definition'''
input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

theta_min = -2*np.pi
theta_max = -theta_min
theta_dot_min = -10.2
theta_dot_max = 10.2
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

dynamics = PendulumCartBroken()
discretization_step_s = 0.001
sample_time_s = 0.01
dynamics_discrete = SampleAndHold(dynamics, sample_time=sample_time_s, discretization_step=discretization_step_s)

''' Generate data by arbitrary sampling of the dynamics'''

store_data_path = 'data/'
# store_net_path  = 'network/'

os.makedirs(store_data_path, exist_ok=True)
# os.makedirs(store_net_path, exist_ok=True)

n_train_data = 10000
is_output_relative = True

train_samples_x = state_space.sample(n_train_data)
train_samples_u = input_space.sample(n_train_data)
train_samples_in = np.concatenate((train_samples_x.T, train_samples_u.T), axis=1)
if is_output_relative:
    train_samples_out = dynamics_discrete(train_samples_x, train_samples_u) - train_samples_x
else:
    train_samples_out = dynamics_discrete(train_samples_x, train_samples_u)

train_samples_out = train_samples_out.T

with open(store_data_path + 'data_in.pickle', 'wb') as handle:
    pickle.dump(train_samples_in, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_data_path + 'data_out.pickle', 'wb') as handle:
    pickle.dump(train_samples_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(store_data_path + 'data_in.pickle', 'rb') as handle:
#     train_samples_in = pickle.load(handle)
#
# with open(store_data_path + 'data_out.pickle', 'rb') as handle:
#     train_samples_out = pickle.load(handle)
#
# '''Fitting scaler and training network'''
#
# scaler_in = preprocessing.StandardScaler()
# scaler_out = preprocessing.StandardScaler()
#
# scaler_in.fit(train_samples_in)
# scaler_out.fit(train_samples_out)
#
# scaler = [scaler_in, scaler_out]
#
# input_size = 3
# block_size = 1
# output_size = 2
# network = PendulumFC(input_size, block_size, output_size)
# lr = 5e-03  # starting learning rate
# wd = 1e-3  # weight decay
# gamma_1 = 0.75  # rate of learning rate adjustment for multisteplr scheduler (time horizon)
# gamma_2 = 0.5  # rate of learning rate adjustment for plateau scheduler
# optimizer = opt.Adam(network.parameters(), lr=lr, weight_decay=wd)
# epochs = 500
#
# network, loss = pendulum_train_explicit_1step(train_samples_in, train_samples_out, optimizer=optimizer, network=network, epochs=epochs, scaler=scaler)
#
# save_network_to_folder(network, scaler=scaler, folder_path= store_net_path, is_output_relative=is_output_relative)

