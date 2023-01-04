from dynamics_library import PendulumCartBroken
from dynamical_rrt import DynamicalRRT
from zonotope_lib import Box
import numpy as np
import pickle

input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

theta_min = -2 * np.pi
theta_max = -theta_min
theta_dot_min = -10.0
theta_dot_max = 10.0
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

root_state = np.array([[0], [0]])

dynamics = PendulumCartBroken()


target_set = Box(np.array([[3.0, 3.2], [-0.001, 0.001]]))

n_trajectories = 10
trace_length = 40
all_traces = np.zeros((trace_length, 3, 0))
for i in range(n_trajectories):
    rrt = DynamicalRRT(dynamics, state_space, input_space, root_state, backward=True, target_region=target_set, nsteps=1)
    rrt.planning(animation=False, max_iter=5000)
    traces = rrt.get_traces(trace_time_length=trace_length, overlapping=False)
    traces = traces.reshape((trace_length, 3, traces.shape[1]))
    all_traces = np.concatenate((all_traces, traces), 2)

with open('broken_pendulum_traces.pickle', 'wb') as handle:
    pickle.dump(all_traces, handle, protocol=pickle.HIGHEST_PROTOCOL)