import matplotlib.pyplot as plt
from dynamics_library import PendulumCartContinuous, PendulumCartBroken, SampleAndHold
from pwa_lib import compute_multistep_affine_dynamics
from zonotope_lib import Box
import numpy as np
from copy import copy

input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

min_u = -15
max_u = 15

theta_min = -0.5
theta_max = -theta_min
theta_dot_min = -.5
theta_dot_max = .5
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

target = np.array([[0], [0]])

dynamics = PendulumCartBroken()
sampling_time = 0.01
discretization_time = 0.001
dynamics_discrete_fwd = SampleAndHold(dynamics, sample_time=sampling_time, discretization_step=discretization_time)
dynamics_discrete_bwd = SampleAndHold(dynamics, sample_time=sampling_time, discretization_step=discretization_time,
                                      backward=True)

'''Controller for local stabilization around target'''
nsteps = 5
input_range = input_space.get_range()
input_space_multistep = Box(np.tile(input_range, (nsteps, 1)))

state = Box(np.concatenate((target, target), 1))
affine_dynamics_bwd = compute_multistep_affine_dynamics(dynamics_discrete_bwd, nsteps, state_box=state,
                                                        input_box=input_space)
reach_set = affine_dynamics_bwd.get_ru(input_space_multistep) + affine_dynamics_bwd.get_rx(state).center
local_ctrl = 2 * reach_set.generators_pinv

def get_controller():
    return local_ctrl

if __name__ == "__main__":
    '''Apply controller from random states'''
    sim_steps = 200
    trials = 50

    first_d_id = 80  # denotes the last dataset gathered
    for trial in range(trials):
        rand_state = state_space.sample()
        crnt_state = copy(rand_state)
        counter = 0
        ctrl_len = 0
        x_hist = crnt_state[0].reshape(-1, 1)
        y_hist = crnt_state[1].reshape(-1, 1)
        u_hist = np.zeros_like(x_hist).reshape(-1, 1)

        for t in range(sim_steps):
            if counter == 0:
                ctrl_input = local_ctrl @ crnt_state
                counter = len(ctrl_input)
                ctrl_len = len(ctrl_input)
            step = counter - 1
            crnt_u = ctrl_input[step]
            if crnt_u > max_u:
                crnt_u = np.array([input_max])
            elif crnt_u < min_u:
                crnt_u = np.array([min_u])

            crnt_state = dynamics_discrete_fwd(x=crnt_state, u=crnt_u)
            x_hist = np.concatenate((x_hist, crnt_state[0].reshape(-1, 1)), axis=0)
            y_hist = np.concatenate((y_hist, crnt_state[1].reshape(-1, 1)), axis=0)
            u_hist = np.concatenate((u_hist, crnt_u.reshape(-1, 1)), axis=0)
            counter -= 1
        dataset = np.concatenate((x_hist, y_hist, u_hist), axis=1)
        np.savetxt(fname='dataset_' + str(trial+first_d_id) + '.txt', X=dataset, delimiter=',')
        plt.plot(x_hist, label=str(trial))
    plt.legend()
    plt.savefig("datasets_"+str(first_d_id)+"_until_"+str(first_d_id+trial)+".svg")
    plt.show()


