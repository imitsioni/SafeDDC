import matplotlib.pyplot as plt

from dynamics_library import PendulumCartBroken, SampleAndHold
from dynamical_rrt import DynamicalRRT
from zonotope_lib import Box
import numpy as np
import pickle
import os

input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

theta_min = -np.pi * 1.0
theta_max = -theta_min
theta_dot_min = -5.0
theta_dot_max = 5.0
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))


def get_distance(x1, x2):
    size = state_space.get_bounding_box_size()
    dist = (np.abs(x1-x2))/size
    return np.sum(dist)

store_data_path = 'data/'
os.makedirs(store_data_path, exist_ok=True)


root_state = np.array([[0], [0]])

dynamics = PendulumCartBroken()
discretization_step_s = 0.001
sample_time_s = 0.01
dynamics_discrete = SampleAndHold(dynamics, sample_time=sample_time_s, discretization_step=discretization_step_s)


n_trajectories = 100
trace_length = 100

train_samples_in = np.zeros((n_trajectories*trace_length, 3))
train_samples_out = np.zeros((n_trajectories*trace_length, 2))

is_output_relative = True


all_traces = np.zeros((trace_length, 3, 0))
cnt = 0
for i in range(n_trajectories):
    x = root_state
    u = 0*input_space.sample()
    for t in range(trace_length):
        x_target = state_space.sample()
        u = input_space.sample(10)
        min_dist = np.inf
        best_x = None
        best_u = None
        for j in range(u.size):
            x_new = dynamics_discrete(x, u[0, j:j+1])
            if get_distance(x_new, x_target)<min_dist:
                min_dist = get_distance(x_new, x_target)
                best_x = x_new
                best_u = u[0, j:j+1]
        train_samples_in[cnt, :] = np.concatenate((x.reshape((1,2)), best_u.reshape((1,1))), axis=1)
        if is_output_relative:
            train_samples_out[cnt, :] = (best_x - x).reshape((1, 2))
        else:
            train_samples_out[cnt, :] = best_x.reshape((1, 2))
        x = best_x
        u = best_u
        cnt+=1

with open(store_data_path + 'data_in.pickle', 'wb') as handle:
    pickle.dump(train_samples_in, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_data_path + 'data_out.pickle', 'wb') as handle:
    pickle.dump(train_samples_out, handle, protocol=pickle.HIGHEST_PROTOCOL)