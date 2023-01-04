import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib
import os
from inverted_pendulum.pendulum_network_interface import simple_train, save_network_to_folder, load_network_from_folder, NetworkDynamics_1step
from inverted_pendulum.zonotope_lib import Box
from inverted_pendulum.dynamics_library import PendulumCartBroken, SampleAndHold

'''PLot properties'''
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 35}

matplotlib.rc('font', **font)



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

nrand = 10000
rand_state = state_space.sample(nrand)
rand_input = input_space.sample(nrand)

actual_out = dynamics_discrete(rand_state, rand_input)
actual_accel = actual_out[1, :] - rand_state[1, :]
actual_vel = actual_out[0, :] - rand_state[0, :]

avg_accel = np.linalg.norm(actual_accel, ord=1) / nrand
avg_vel = np.linalg.norm(actual_vel, ord=1) / nrand


dist_folders = ['localrand_pi', 'localrand_0', 'rrtc', 'rrtc+rand', 'fullrand']
label_dict = {'localrand_0': r"RW $\theta = 0$", 'localrand_pi': r"RW $\theta = \pi$", \
    'rrtc':'RRTC', 'rrtc+rand': 'RRTC + rand', 'fullrand': \
              'FRAND'}
error_table = np.zeros((0, 2))
for df in dist_folders:
    with open(df + '/data/data_in.pickle', 'rb') as handle:
        data_in = pickle.load(handle)
    with open(df + '/data/data_out.pickle', 'rb') as handle:
        data_out = pickle.load(handle)

    net_dir = df + '/net'
    os.makedirs(net_dir, exist_ok=True)

    if not os.path.isfile(net_dir + '/info.xls'):
        network, scaler = simple_train(data_in, data_out, epochs=500)
        save_network_to_folder(network, scaler, is_output_relative=True, folder_path=net_dir+'/')

    network, scaler, info = load_network_from_folder(net_dir + '/')

    network_dynamics = NetworkDynamics_1step(network, scaler, net_type=info['net_type'][0], is_output_relative=info['is_output_relative'][0])
    net_out = network_dynamics(rand_state, rand_input)
    pred_accel = net_out[1,:] - rand_state[1,:]
    pred_vel = net_out[0, :] - rand_state[0, :]

    accel_err = np.abs(actual_accel-pred_accel)
    vel_err = np.abs(actual_vel-pred_vel)
    error_table = np.concatenate((error_table, np.array([[np.mean(accel_err), np.std(accel_err) / 2]])))

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = 'Latin Modern Math'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 13
plt.errorbar(range(len(dist_folders)), error_table[:, 0], error_table[:, 1], fmt='o', color='black',
             ecolor='lightgray', elinewidth=50, capsize=0)
ax = plt.gca()
ax.set_xticks(range(len(dist_folders)))
ax.set_xticklabels([label_dict[df] for df in dist_folders])
ax.set_title('Acceleration prediction error')
ax.grid('on')
# plt.show()
plt.savefig('acceleration_errors.svg', bbox_inches='tight')