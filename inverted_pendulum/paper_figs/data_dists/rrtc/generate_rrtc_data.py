from dynamical_rrt import DynamicalRRT
from dynamics_library import PendulumCartBroken, SampleAndHold
import pickle
import matplotlib.pyplot as plt
from zonotope_lib import Box
import numpy as np
import os

store_data_path = 'data/'
os.makedirs(store_data_path, exist_ok=True)

with open('tree_10000_rewired_input10_doublefriction.pickle', 'rb') as f:
    rrt = pickle.load(f)
assert(isinstance(rrt, DynamicalRRT))

dynamics = PendulumCartBroken()
discretization_step_s = 0.001
sample_time_s = 0.01
dynamics_discrete = SampleAndHold(dynamics, sample_time=sample_time_s, discretization_step=discretization_step_s)

input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

theta_min = -2*np.pi
theta_max = -theta_min
theta_dot_min = -10.2
theta_dot_max = 10.2
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))


root_state = np.array([[0], [0]])

target_set = Box(np.array([[100.0, 100.0], [100.0, 100.0]]))

n_trajectories = 50
trace_length = 200
random_block_length = 10
random_block_rate = 0.0
play_time = 100
ctrl_full_horizon = True

train_samples_in = np.zeros((n_trajectories*trace_length, 3))
train_samples_out = np.zeros((n_trajectories*trace_length, 2))

is_output_relative = True
cnt = 0
for i in range(n_trajectories):
    x = state_space.sample()
    t = 0
    u = np.array([[]])
    for t in range(trace_length):
        if u.shape[1] == 0:
            if np.random.rand() < random_block_rate and t < play_time:
                is_random = True
                u = input_space.sample(random_block_length).T
            else:
                is_random = False
                u = rrt.get_control(x)

        new_x = dynamics_discrete(x, u[0])

        train_samples_in[cnt, :] = np.concatenate((x.reshape((1, 2)), u[0].reshape((1, 1))), axis=1)
        if is_output_relative:
            train_samples_out[cnt, :] = (new_x - x).reshape((1, 2))
        else:
            train_samples_out[cnt, :] = new_x.reshape((1, 2))

        x = new_x
        cnt+=1

        if ctrl_full_horizon or is_random:
            u = np.delete(u, 0, axis=1)
        else:
            u = np.array([[]])

with open(store_data_path + 'data_in.pickle', 'wb') as handle:
    pickle.dump(train_samples_in, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_data_path + 'data_out.pickle', 'wb') as handle:
    pickle.dump(train_samples_out, handle, protocol=pickle.HIGHEST_PROTOCOL)