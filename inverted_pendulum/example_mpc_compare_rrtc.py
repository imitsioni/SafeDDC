import matplotlib.pyplot as plt
from pendulum_mpc import *
import numpy as np
from copy import copy
from dynamical_rrt import DynamicalRRT
from example_multistep_feedback_bounded import get_controller


class GetMPCInput:
    def __init__(self, safety_on=True, rrtc_cost=True):
        self.feasible_inputs = generate_inputs(N_FSB_INPS, BLOCK_SIZE)
        self.safety_on = safety_on
        self.rrtc_cost = rrtc_cost
        self.state_filler = np.zeros(
            (BLOCK_SIZE, 1))  # used to bring the predictions to the right shape for inverse scaling

    def __call__(self, state_buffer):
        best_so_far = 1e03
        best_move = np.zeros([1, 1])
        best_prediction = []
        # Do the bestest MPC in the universe
        costs = []
        input_safety_buffer = np.zeros(len(self.feasible_inputs))

        for idx, inp in enumerate(self.feasible_inputs):
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
                    next_state_sc = np.concatenate((next_state, self.state_filler), axis=1)
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
                if self.safety_on:
                    distance = check_safety(torch.tensor(current_net_input_scaled), scaled_next_state,
                                            encoder, safeset)
                else:
                    distance = None

                cost = cost_function(unscaled_next_state, inp, distance, self.rrtc_cost)
                if cost < best_so_far:
                    best_move = np.copy(inp)
                    best_prediction = np.copy(unscaled_next_state)
                    best_ind = idx
                    best_so_far = cost
        return best_move, best_so_far, best_prediction


def run_mpc(x_0: np.ndarray, time_steps: int, get_mpc_input: GetMPCInput, trial: int, name: str):
    # Fill a buffer with 'BLOCK_SIZE' states for the first network input
    state_buffer = x_0.reshape(1, -1)
    # Do nothing until we fill the buffer
    crnt_state = dynamics_discrete_fwd(x=state_buffer.transpose(), u=np.zeros([1, 1]))
    while len(state_buffer) < BLOCK_SIZE:
        state_buffer = np.concatenate((state_buffer, crnt_state.reshape(1, -1)), axis=0)
        crnt_state = dynamics_discrete_fwd(x=crnt_state, u=np.zeros([1, 1]))
    # Initialize the buffer for safety trends. The buffer contains blocks of states+inputs
    safety_distance_buffer_temp = np.copy(state_buffer)
    while len(safety_distance_buffer_temp) < (HRZN + 1) * BLOCK_SIZE:
        safety_distance_buffer_temp = np.concatenate((safety_distance_buffer_temp, crnt_state.reshape(1, -1)), axis=0)
        crnt_state = dynamics_discrete_fwd(x=crnt_state, u=np.zeros([1, 1]))
    safety_distance_buffer_sc = np.concatenate((safety_distance_buffer_temp, np.ones(((HRZN + 1) * BLOCK_SIZE, 1))),
                                               axis=1)
    safety_distance_buffer_sc = scaler.transform(safety_distance_buffer_sc)
    safety_distance_buffer = np.copy(safety_distance_buffer_sc[:, :OUTP_SIZ])
    # Initialize placeholders for plotting
    x_hist = crnt_state[0].reshape(-1, 1)
    y_hist = crnt_state[1].reshape(-1, 1)
    u_hist = np.zeros_like(x_hist).reshape(-1, 1)

    # Initialize the counter for safety trends. If the system has not moved closer to the safe set in the last
    # SFT_STPS, the trial is aborted
    safety_trend_counter = 0

    safe_trial = True if get_mpc_input.safety_on else None
    # Initialize the distance to the safe set as far
    dist_to_safe_set = -1.0
    first_time = True
    for t in range(time_steps):
        best_move, best_cost, best_prediction = get_mpc_input(state_buffer)
        crnt_state = dynamics_discrete_fwd(x=crnt_state, u=best_move)
        crnt_state_scaled = np.concatenate((crnt_state.reshape(1, -1), best_move.reshape(1, -1)), axis=1)
        crnt_state_scaled = scaler.transform(crnt_state_scaled)[:, :OUTP_SIZ]
        if get_mpc_input.safety_on:
            safety_distance_buffer = np.concatenate((safety_distance_buffer[1:, :], crnt_state_scaled.reshape(1, -1)),
                                                    axis=0)

            actual_distance_to_safe_set = check_safety(torch.tensor(safety_distance_buffer[:1, :]),
                                                       torch.tensor(safety_distance_buffer[1:, :]),
                                                       encoder, safeset)
            # print("Checking if ", actual_distance_to_safe_set, "is smaller than ", dist_to_safe_set)
            if np.sign(actual_distance_to_safe_set) < 0 and actual_distance_to_safe_set < dist_to_safe_set:
                safety_trend_counter += 1
                if safety_trend_counter > SFT_STPS:
                    # print(" Step", t, "we are too far with ", actual_distance_to_safe_set, dist_to_safe_set, safety_trend_counter)
                    if first_time: print("Aborting trial ", trial, " at step ", t)
                    first_time = False
                    safe_trial = False
            elif np.sign(actual_distance_to_safe_set) < 0 and actual_distance_to_safe_set > dist_to_safe_set:
                # print("Current distance is ", actual_distance_to_safe_set)
                dist_to_safe_set = actual_distance_to_safe_set
                safety_trend_counter = 0
            elif np.sign(actual_distance_to_safe_set):
                safety_trend_counter = 0
        # print("Step ", t, "distance ", actual_distance_to_safe_set)
        # Logging the collected datasets
        x_hist = np.concatenate((x_hist, crnt_state[0].reshape(-1, 1)), axis=0)
        y_hist = np.concatenate((y_hist, crnt_state[1].reshape(-1, 1)), axis=0)
        u_hist = np.concatenate((u_hist, best_move[0, :].reshape(-1, 1)), axis=0)
        state_buffer = np.concatenate((state_buffer[1:, :], crnt_state.reshape(1, -1)), axis=0)
    dataset = np.concatenate((x_hist, y_hist, u_hist), axis=1)
    np.savetxt(fname="data/mpc_data/generalization_compare_mpc_" + str(name) + '_trial_' + str(trial + FRST_DID) + ".txt", X=dataset,
               delimiter=',')
    return {'x_hist': x_hist, 'y_hist': y_hist, 'u_hist': u_hist, 'state_buffer': state_buffer,
            'safety_distance_buffer': safety_distance_buffer, 'safe_trial': safe_trial}


def run_rrtc(x_0: np.ndarray, time_steps: int, tree: DynamicalRRT):
    x_hist = np.zeros((0, 1))
    y_hist = np.zeros((0, 1))
    u_hist = np.zeros((0, 1))

    x = copy(x_0)
    cnt = 0
    while True:
        if cnt >= time_steps:
            break
        u = tree.get_control(x)
        for i in range(len(u)):
            x_hist = np.concatenate((x_hist, x[0].reshape((1, 1))), axis=0)
            y_hist = np.concatenate((y_hist, x[1].reshape((1, 1))), axis=0)
            u_hist = np.concatenate((u_hist, u), axis=0)
            x = dynamics_discrete_fwd(x, u[i])
            cnt += 1

    return {'x_hist': x_hist, 'y_hist': y_hist, 'u_hist': u_hist}


def run_multistep(x_0: np.ndarray, time_steps: int, controller: np.ndarray, u_min=-15.0, u_max=15.0):
    x_hist = np.zeros((0, 1))
    y_hist = np.zeros((0, 1))
    u_hist = np.zeros((0, 1))

    x = copy(x_0)
    cnt = 0
    ctrl = np.array([[]])
    while True:
        if cnt >= time_steps:
            break
        u = controller @ x
        for i in range(len(u)):
            uu = u[len(u) - i - 1]
            if uu > u_max:
                uu = np.array([u_max])
            elif uu < u_min:
                uu = np.array([u_min])

            x_hist = np.concatenate((x_hist, x[0].reshape((1, 1))), axis=0)
            y_hist = np.concatenate((y_hist, x[1].reshape((1, 1))), axis=0)
            u_hist = np.concatenate((u_hist, uu.reshape(1, 1)), axis=0)
            x = dynamics_discrete_fwd(x, uu)
            cnt += 1

    return {'x_hist': x_hist, 'y_hist': y_hist, 'u_hist': u_hist}


def check_converge(x_trace, tolerance):
    tail_len = 20
    final_err = np.abs(x_trace[-tail_len:])
    return all(final_err < tolerance)


state_space, input_space, dynamics_discrete_fwd, dynamics_discrete_bwd = initialize_dynamics(input_min=INP_MIN,
                                                                                             theta_min=TH_MIN,
                                                                                             theta_dot_min=TH_D_MIN)  # Initialize the dynamics network
network0, scaler0, info = load_network_from_folder('paper_figs/data_dists/rrtc/net/')
if isinstance(scaler0, list):
    scaler_delta0 = scaler0[1]
    scaler0 = scaler0[0]
network1, scaler1, info = load_network_from_folder('paper_figs/data_dists/fullrand/net/')
if isinstance(scaler1, list):
    scaler_delta1 = scaler1[1]
    scaler1 = scaler1[0]
network2, scaler2, info = load_network_from_folder('rrtc+random/converged_trials/relative_net/')
if isinstance(scaler2, list):
    scaler_delta2 = scaler2[1]
    scaler2 = scaler2[0]

networks = {'rrtc': [network0, scaler0, scaler_delta0],
            'frand': [network1, scaler1, scaler_delta1],
            'rrtc+rand': [network2, scaler2, scaler_delta2]
            }

# Initialize the encoder
encoder = initialize_encoder(trained_model=AE)
# Initialize the safe set approximation
safeset = load(PLG)

# Initialize the simulation
first_d_id = FRST_DID  # denotes the last dataset gathered, for figure naming purposes

'''Setting up configurations'''
# Only safety on/off for now
trials_per_config = 100
config1 = {'model': 'rrtc', 'SAFETY_ON': False, 'RRTC_cost': False}
config2 = {'model': 'frand', 'SAFETY_ON': False, 'RRTC_cost': False}
config3 = {'model': 'rrtc+rand', 'SAFETY_ON': False, 'RRTC_cost': False}
config4 = {'model': 'rrtc+rand', 'SAFETY_ON': True, 'RRTC_cost': False}
config5 = {'model': 'rrtc+rand', 'SAFETY_ON': False, 'RRTC_cost': True}
config6 = {'model': 'rrtc+rand', 'SAFETY_ON': True, 'RRTC_cost': True}

configs = [config1, config2, config3, config4, config5, config6]

if FIX_INTIAL_ST:
    starting_states = INIT_STATES[:, :trials_per_config]
else:
    starting_states = state_space.sample(trials_per_config)

for conf in configs:
    network = networks[conf['model']][0]
    scaler = networks[conf['model']][1]
    scaler_delta = networks[conf['model']][2]
    get_mpc_input = GetMPCInput(safety_on=conf['SAFETY_ON'], rrtc_cost=conf['RRTC_cost'])
    tot_converged = 0
    tot_safe = 0
    for trials in range(trials_per_config):
        history = run_mpc(x_0=starting_states[:, trials], time_steps=SIM_STPS, get_mpc_input=get_mpc_input,
                          trial=trials, name=conf['model'] + '_safetyON_' + str(conf['SAFETY_ON']) + '_rrt_cost_' + str(
                conf['RRTC_cost']))
        if trials_per_config <= 10:
            plt.plot(history['x_hist'],
                     label="MPC dataset_" + str(first_d_id + trials) + "_safe_" + str(history['safe_trial']))
        has_converged = check_converge(history['x_hist'], tolerance=0.2)
        tot_converged += has_converged
        if history['safe_trial']: tot_safe += 1

    print(conf)
    print(str(tot_converged) + ' converged from total ' + str(trials_per_config) + ' trials.')
    print(str(tot_safe) + ' were judged safe from total ' + str(trials_per_config) + ' trials.')
    if trials_per_config <= 10:
        plt.title(conf)
        plt.legend()
        plt.show()

'''add one more attempt from the rrtc'''
tot_converged_rrtc = 0
with open('tree_10000_rewired_input10_doublefriction.pickle', 'rb') as f:
    rrt = pickle.load(f)
assert (isinstance(rrt, DynamicalRRT))
fig, ax = plt.subplots()
for trials in range(trials_per_config):
    history = run_rrtc(x_0=starting_states[:, trials:trials + 1], time_steps=SIM_STPS, tree=rrt)
    if trials_per_config <= 10:
        plt.plot(history['x_hist'],
                 label="RRTC dataset_" + str(first_d_id + trials))
    has_converged = check_converge(history['x_hist'], tolerance=0.2)
    tot_converged_rrtc += has_converged
print('RRTC:' + str(tot_converged_rrtc) + ' converged from total ' + str(trials_per_config) + ' trials.')
if trials_per_config <= 10:
    plt.title('RRTC')
    plt.legend()
    plt.show()

'''Add another attemp with the bounded multistep controller'''
# fig, ax = plt.subplots()
multi_step_controller = get_controller()
tot_converged_multistep = 0
for trials in range(trials_per_config):
    history = run_multistep(x_0=starting_states[:, trials:trials + 1], time_steps=SIM_STPS,
                            controller=multi_step_controller)
    if trials_per_config <= 10:
        plt.plot(history['x_hist'],
                 label="multistep dataset_" + str(first_d_id + trials))
    has_converged = check_converge(history['x_hist'], tolerance=0.2)
    tot_converged_multistep += has_converged
print(str(tot_converged_multistep) + ' converged from total ' + str(trials_per_config) + ' trials.')
if trials_per_config <= 10:
    plt.title('Multistep')
    plt.legend()
    plt.show()




# a = plt.legend()
# # plt.gcf().set_size_inches(10, 7)
# plt.savefig("data/mpc_data/mpc_compare_datasets_" + str(first_d_id) + "_until_" + str(first_d_id + trials) + ".svg")
# plt.savefig("data/mpc_data/mpc_compare_datasets_" + str(first_d_id) + "_until_" + str(first_d_id + trials) + ".png")


# plt.show()
#
# #save the legend separately
# a2 = a.figure
# a2.canvas.draw()
# bbox = a.get_window_extent().transformed(a2.dpi_scale_trans.inverted())
# a2.savefig("data/mpc_data/legend_mpc_datasets_" + str(first_d_id) + "_until_" + str(first_d_id + trials)+".png",
#            dpi="figure", bbox_inches=bbox)
