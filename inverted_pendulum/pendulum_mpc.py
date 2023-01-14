import matplotlib.pyplot as plt
from dynamics_library import PendulumCartBroken, SampleAndHold
from zonotope_lib import Box
import numpy as np
from pendulum_network_definitions import PendulumVariationalEncoderMultistep
import torch
from joblib import  load

from exp_config import *  # All constants are in here
import shapely.geometry as sh
import pickle
from safety.safeset import *
from pendulum_network_interface import load_network_from_folder

with open('generated_trees/tree_15000_rewired_input10_doublefriction.pickle', 'rb') as f:
    rrt = pickle.load(f)

def initialize_dynamics(input_min, theta_min, theta_dot_min, target=np.array([[0], [0]])):
    # Initialize the state-input space
    input_max = -input_min
    input_space = Box(np.array([[input_min, input_max]]))

    theta_max = -theta_min
    theta_dot_max = -theta_dot_min
    state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

    # Initialize the dynamics of the pendulum
    dynamics = PendulumCartBroken()
    sampling_time = 0.01
    discretization_time = 0.001
    dynamics_discrete_fwd = SampleAndHold(dynamics, sample_time=sampling_time, discretization_step=discretization_time)
    dynamics_discrete_bwd = SampleAndHold(dynamics, sample_time=sampling_time, discretization_step=discretization_time,
                                          backward=True)
    return state_space, input_space, dynamics_discrete_fwd, dynamics_discrete_bwd


def generate_inputs(N_FSB_INPS, block_size):
    feas_inp = np.linspace(FSB_INP_MIN, -FSB_INP_MIN, N_FSB_INPS).reshape(-1, 1)  # K random forces
    feasible_inputs = np.zeros((N_FSB_INPS, block_size, feas_inp.shape[1]))
    for i in range(len(feas_inp)):
        force_input = torch.ones(block_size, 1)
        force_input[:, 0] = force_input[:, 0] * feas_inp[i]
        feasible_inputs[i] = force_input
    return feasible_inputs


def initialize_encoder(trained_model):
    trained_ae = torch.load(trained_model)
    trained_ae_items = list(trained_ae.items())
    # Load the first weights that correspond to the encoder
    enc = PendulumVariationalEncoderMultistep(HRZN, BLOCK_SIZE)
    encoder_state_dict = enc.state_dict()
    count = 0
    for key, value in encoder_state_dict.items():
        layer_name, weights = trained_ae_items[count]
        encoder_state_dict[key] = weights
        count += 1
    enc.load_state_dict(encoder_state_dict)
    return enc


def check_safety(current_state, predicted_state, encoder, safeset):
    x_latent = encoder(current_state[None, :, :OUTP_SIZ].float(),
                       predicted_state.reshape(HRZN, BLOCK_SIZE, -1).float())
    # If using the VAE, keep only the mean
    if len(x_latent) > 1: x_latent = x_latent[0]
    _, pred = safeset.query_safety(np.array(x_latent.detach()))
    distance = np.copy(pred)
    return distance


def cost_function(unscaled_prediction, input, distance=None, rrtc_cost=False):
    c_pos = 0.5
    c_vel = 0.0
    c_inp = 0.0
    c_dis = 0.1
    c_rrt = 1.0
    # fixme you can potentially have a steeper exp, e.g. np.exp(-2*distance) for different behaviors
    cost = c_pos * np.linalg.norm(unscaled_prediction[:, 0]) \
           + c_vel * np.linalg.norm(unscaled_prediction[:, 1]) \
           + c_inp * np.linalg.norm(input)
    if distance is not None:
        cost += c_dis * (np.exp(-min(0, distance))-1)
        # cost += c_dis * np.exp(-distance)
    if rrtc_cost:
        cost += + c_rrt * rrt.get_state_cost(np.array(unscaled_prediction[-1:, :2]).T)

    return cost


if __name__ == "__main__":
    # Initialize the state-action space
    state_space, input_space, dynamics_discrete_fwd, dynamics_discrete_bwd = initialize_dynamics(input_min=INP_MIN,
                                                                                                 theta_min=TH_MIN,
                                                                                                 theta_dot_min=TH_D_MIN)
    # Initialize the dynamics network
    network, scaler, info = load_network_from_folder(DNM_MDL)
    if isinstance(scaler, list):
        scaler_delta = scaler[1]
        scaler = scaler[0]

    # Initialize the encoder
    encoder = initialize_encoder(trained_model=AE)
    # Initialize the safe set approximation
    safeset = load(PLG)

    # Initialize the simulation
    first_d_id = FRST_DID  # denotes the last dataset gathered, for figure naming purposes
    state_filler = np.zeros((BLOCK_SIZE, 1))  # used to bring the predictions to the right shape for inverse scaling

    for trial in range(TRLS):
        # Fill a buffer with 'BLOCK_SIZE' states for the first network input
        if FIX_INTIAL_ST:
            state_buffer = INIT_STATES[:, trial].reshape(1, -1)
        else:
            state_buffer = state_space.sample().reshape(1, -1)
        # Do nothing until we fill the buffer
        crnt_state = dynamics_discrete_fwd(x=state_buffer.transpose(), u=np.zeros([1, 1]))
        while len(state_buffer) < BLOCK_SIZE:
            state_buffer = np.concatenate((state_buffer, crnt_state.reshape(1, -1)), axis=0)
            crnt_state = dynamics_discrete_fwd(x=crnt_state, u=np.zeros([1, 1]))
        # Initialize the buffer for safety trends. The buffer contains blocks of states+inputs
        safety_distance_buffer_temp = np.copy(state_buffer)
        while len(safety_distance_buffer_temp) < (HRZN + 1) * BLOCK_SIZE:
            safety_distance_buffer_temp = np.concatenate((safety_distance_buffer_temp, crnt_state.reshape(1, -1)),
                                                         axis=0)
            crnt_state = dynamics_discrete_fwd(x=crnt_state, u=np.zeros([1, 1]))
        safety_distance_buffer_sc = np.concatenate((safety_distance_buffer_temp, np.ones(((HRZN + 1) * BLOCK_SIZE, 1))),
                                                   axis=1)
        safety_distance_buffer_sc = scaler.transform(safety_distance_buffer_sc)
        safety_distance_buffer = np.copy(safety_distance_buffer_sc[:, :OUTP_SIZ])

        # Initialize placeholders for plotting
        # ToDo: Move these descriptions to the docstring so they're visible
        # 1. x_hist contains the actual positions of the system
        x_hist = crnt_state[0].reshape(-1, 1)
        # 2. x_hist contains the actual velocities of the system
        y_hist = crnt_state[1].reshape(-1, 1)
        # 3. u_hist contains the MPC inputs on the system
        u_hist = np.zeros_like(x_hist).reshape(-1, 1)
        # 4. d_hist contains the PREDICTED distances from the safe set
        d_hist = np.zeros((1, 1))
        # 4. actual_d_hist contains the actual distances from the safe set
        actual_d_hist = np.zeros((0, 1))

        # Initialize the counter for safety trends. If the system has not moved closer to the safe set in the last
        # SFT_STPS, the trial is aborted
        safety_trend_counter = 0
        safe_trial = True

        # Initialize the distance to the safe set as far
        dist_to_safe_set = -1.0
        # This is where the MPC iterations start
        for t in range(SIM_STPS):
            best_so_far = 1e03
            best_move = np.zeros([1, 1])
            best_prediction = []
            # Generate new inputs
            feasible_inputs = generate_inputs(N_FSB_INPS, BLOCK_SIZE)
            # Do the bestest MPC in the universe
            costs = []
            input_safety_buffer = np.zeros(len(feasible_inputs))
            dsts = []
            for idx, inp in enumerate(feasible_inputs):
                with torch.no_grad():
                    next_state = np.copy(state_buffer)
                    current_net_input = np.concatenate((state_buffer, inp), axis=1)
                    current_net_input_scaled = scaler.transform(current_net_input)
                    unscaled_next_state = torch.zeros((0, 3))
                    scaled_next_state = torch.zeros((0, OUTP_SIZ))
                    iterative_in = torch.tensor(current_net_input_scaled).float()
                    for bl_ind in range(1, HRZN + 1):
                        out = network(iterative_in)  # scaled delta
                        unscaled_delta = scaler_delta.inverse_transform(out.detach().view(BLOCK_SIZE, OUTP_SIZ))
                        next_state = unscaled_delta + next_state[-1, :OUTP_SIZ]  # unscaled abs
                        next_state_sc = np.concatenate((next_state, state_filler), axis=1)
                        next_state_sc = scaler.transform(next_state_sc)  # absolute and scaled
                        # Keep the scaled prediction in a buffer so it can be used with the OOD detector
                        scaled_next_state = torch.cat(
                            (scaled_next_state, torch.tensor(next_state_sc[:, :OUTP_SIZ])))
                        full_next_state = np.concatenate((next_state, inp), axis=1)
                        # Keep the unscaled predictions in a buffer so we can calculate the cost
                        unscaled_next_state = torch.cat((unscaled_next_state, torch.tensor(full_next_state)))
                        # Use the scaled previous prediction as the new input and add the scaled control input
                        iterative_in = torch.cat((torch.tensor(next_state_sc[:, :OUTP_SIZ]),
                                                  torch.tensor(current_net_input_scaled[:, OUTP_SIZ:])), 1).float()

                    # Check the safety state
                    distance = check_safety(torch.tensor(current_net_input_scaled), scaled_next_state,
                                            encoder, safeset)
                    cost = cost_function(unscaled_next_state, inp, distance)
                    # dsts is initialized every simulation step so it is easier to see what score the best move got
                    dsts.append(distance)
                    if cost < best_so_far:
                        best_move = np.copy(inp)
                        best_prediction = np.copy(unscaled_next_state)
                        best_ind = idx
                        best_so_far = cost
            crnt_state = dynamics_discrete_fwd(x=crnt_state, u=feasible_inputs[best_ind, 0, :])
            crnt_state_scaled = np.concatenate((crnt_state.reshape(1, -1), best_move.reshape(1, -1)), axis=1)
            crnt_state_scaled = scaler.transform(crnt_state_scaled)[:, :OUTP_SIZ]
            safety_distance_buffer = np.concatenate((safety_distance_buffer[1:, :], crnt_state_scaled.reshape(1, -1)), axis=0)

            actual_distance_to_safe_set = check_safety(torch.tensor(safety_distance_buffer[:1, :]),
                                                       torch.tensor(safety_distance_buffer[1:, :]),
                                                       encoder, safeset)
            # print("Checking if ", actual_distance_to_safe_set, "is smaller than ", dist_to_safe_set)
            if np.sign(actual_distance_to_safe_set) < 0 and actual_distance_to_safe_set < dist_to_safe_set:
                safety_trend_counter += 1
                if safety_trend_counter > SFT_STPS:
                    # print("we are too far with ", actual_distance_to_safe_set, dist_to_safe_set)
                    # print("Aborting trial ", trial, " at step ", t)
                    # break
                    safe_trial = False
            elif np.sign(actual_distance_to_safe_set) < 0 and actual_distance_to_safe_set > dist_to_safe_set:
                # print("Current distance is ", actual_distance_to_safe_set)
                dist_to_safe_set = actual_distance_to_safe_set
                safety_trend_counter = 0
            elif np.sign(actual_distance_to_safe_set):
                safety_trend_counter = 0
                # print("The new distance is ", dist_to_safe_set)
            # Logging the collected datasets
            x_hist = np.concatenate((x_hist, crnt_state[0].reshape(-1, 1)), axis=0)
            y_hist = np.concatenate((y_hist, crnt_state[1].reshape(-1, 1)), axis=0)
            u_hist = np.concatenate((u_hist, best_move[0, :].reshape(-1, 1)), axis=0)
            d_hist = np.concatenate((d_hist, dsts[best_ind].reshape(-1, 1)), axis=0)
            actual_d_hist = np.concatenate((actual_d_hist, actual_distance_to_safe_set.reshape(-1, 1)), axis=0)
            state_buffer = np.concatenate((state_buffer[1:, :], crnt_state.reshape(1, -1)), axis=0)
        dataset = np.concatenate((x_hist, y_hist, u_hist, d_hist), axis=1)
        np.savetxt(fname="data/mpc_data/dataset_mpc_" + str(trial + first_d_id) + ".txt", X=dataset, delimiter=',')
        plt.plot(x_hist, label="MPC dataset_" + str(first_d_id + trial) + "_safe_" + str(safe_trial))
        print("***** Trial ", trial, ". Mean distances, std, max, min", np.mean(d_hist),
              np.std(d_hist), max(d_hist[:, 0]), min(d_hist[:, 0]))
    plt.legend()
    plt.gcf().set_size_inches(10, 7)
    plt.savefig("data/mpc_data/mpc_datasets_" + str(first_d_id) + "_until_" + str(first_d_id + trial) + ".svg")
    plt.savefig("data/mpc_data/mpc_datasets_" + str(first_d_id) + "_until_" + str(first_d_id + trial) + ".png")

plt.show()
