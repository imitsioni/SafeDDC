import numpy as np
import torch
from inverted_pendulum.pendulum_network_definitions import PendulumFC, PendulumLSTM, PendulumVariationalEncoderMultistep
from joblib import dump, load
import os
import pandas as pd
import sklearn.preprocessing as preprocessing
import torch.optim as opt
from inverted_pendulum.pendulum_train_utils import pendulum_train_explicit_1step

class NetworkDynamics:
    def __init__(self, network, scaler, net_type='lstm', is_output_relative=True, block_size=5):
        self.network = network
        self.net_type = net_type
        self.is_output_relative = is_output_relative
        if is_output_relative or isinstance(scaler, list):
            self.scaler = scaler[0]
            self.scaler_inp = scaler[0]
            self.scaler_lab = scaler[1]
        else:
            self.scaler = scaler
            self.scaler_inp = scaler

        self.block_size = block_size
        self.net_hidden = None
        if net_type == 'lstm':
            self.reset_hidden()

    def __call__(self, x: np.ndarray, u: np.ndarray, return_path=True):
        if x.ndim < 2:
            x = x.reshape((x.size, 1))
        if u.ndim < 2:
            u = u.reshape((u.size, 1))

        if x.shape[1] == 1:
            x = np.tile(x, (1, self.block_size))
        if u.shape[1] == 1:
            u = np.tile(u, (1, self.block_size))
        x = x.T
        u = u.T
        net_in_unscaled = np.concatenate((x, u), axis=1)
        net_in_scaled = self.scaler_inp.transform(net_in_unscaled)
        net_in_tensor = torch.tensor(net_in_scaled).float()
        with torch.no_grad():
            if self.net_type == 'lstm':
                out_diff_scaled, self.net_hidden = self.network(net_in_tensor, self.net_hidden)
            elif self.net_type == 'fc':
                out_diff_scaled = self.network(net_in_tensor)
            else:
                print('unknown network type!')
                return
        if self.is_output_relative:
            # out_diff_scaled = out_diff_scaled.view(self.block_size, -1).detach().numpy()
            out_diff_unscaled = self.scaler_lab.inverse_transform(out_diff_scaled)
            out_diff_unscaled = out_diff_unscaled[:, 0:x.shape[1]]
            out_unscaled = out_diff_unscaled + x[-1, :]
        else:
            state_filler = np.zeros((self.block_size, u.shape[1]))
            out_diff_scaled = np.concatenate(
                (out_diff_scaled.view(self.block_size, -1).detach().numpy(), state_filler),
                axis=1)
            out_diff_unscaled = self.scaler.inverse_transform(out_diff_scaled)
            out_unscaled = out_diff_unscaled[:, 0:x.shape[1]]

        if return_path:
            return out_unscaled[-1, :].reshape((x.shape[1], 1)), out_unscaled
        else:
            return out_unscaled[-1, :].reshape((x.shape[1], 1))

class NetworkDynamics_1step:
    def __init__(self, network, scaler, net_type='lstm', is_output_relative=True):
        self.network = network
        self.net_type = net_type
        self.is_output_relative = is_output_relative
        if is_output_relative or isinstance(scaler, list):
            self.scaler = scaler[0]
            self.scaler_inp = scaler[0]
            self.scaler_lab = scaler[1]
        else:
            self.scaler = scaler
            self.scaler_inp = scaler

        self.net_hidden = None
        if net_type == 'lstm':
            self.reset_hidden()

    def __call__(self, x: np.ndarray, u: np.ndarray):
        if x.ndim < 2:
            x = x.reshape((x.size, 1))
        if u.ndim < 2:
            u = u.reshape((u.size, 1))
        x = x.T
        u = u.T
        net_in_unscaled = np.concatenate((x, u), axis=1)
        net_in_scaled = self.scaler_inp.transform(net_in_unscaled)
        net_in_tensor = torch.tensor(net_in_scaled).float()

        with torch.no_grad():
            if self.net_type == 'lstm':
                out_diff_scaled, self.net_hidden = self.network(net_in_tensor, self.net_hidden)
            elif self.net_type == 'fc':
                out_diff_scaled = self.network(net_in_tensor)
            else:
                print('unknown network type!')
                return
        if self.is_output_relative:
            out_diff_unscaled = self.scaler_lab.inverse_transform(out_diff_scaled)
            out_diff_unscaled = out_diff_unscaled[:, 0:x.shape[1]]
            out_unscaled = out_diff_unscaled + x
        else:
            state_filler = np.zeros((self.block_size, u.shape[1]))
            out_diff_scaled = np.concatenate(
                (out_diff_scaled.view(self.block_size, -1).detach().numpy(), state_filler),
                axis=1)
            out_diff_unscaled = self.scaler.inverse_transform(out_diff_scaled)
            out_unscaled = out_diff_unscaled[:, 0:x.shape[1]]

        return out_unscaled.T

    def check_net(self):
        for name, param in self.network.named_parameters():
            print('Reused network parameters', torch.mean(param), name)

    def reset_hidden(self):
        self.net_hidden = self.network.init_hidden()


class InverseDynamics:
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def __call__(self, x, u, return_path = False):
        if return_path:
            x_new, path = self.dynamics(x, u, return_path)
            return 2 * x - x_new, path
        else:
            x_new = self.dynamics(x, u, return_path)
            return 2*x - x_new


def initialize_network(input_size, output_size, trained_model, block_size=10, net_type='fc'):
    if net_type == 'fc':
        network = PendulumFC(input_size, block_size, output_size)
    elif net_type == 'lstm':
        network = PendulumLSTM(input_size, input_size, block_size)
    else:
        Warning('unknown network type')
        return
    pretrained_dict = torch.load(trained_model)
    network.load_state_dict(pretrained_dict)
    return network


def save_network_to_folder(network, scaler, net_type='fc', block_size=1, output_size=2, is_output_relative=True, folder_path=None):
    os.makedirs(folder_path, exist_ok=True)
    torch.save(network.state_dict(), folder_path+'network')
    if isinstance(scaler, list):
        dump(scaler[0], folder_path + 'pendulum_scaler_inp.joblib')
        dump(scaler[1], folder_path + 'pendulum_scaler_lab.joblib')
    else:
        dump(scaler, folder_path + 'pendulum_scaler.joblib')

    info = pd.DataFrame({'net_type': [net_type], 'block_size': [block_size], 'output_size': [output_size], 'is_output_relative': [is_output_relative]})
    info.to_excel(folder_path+'info.xls')


def load_network_from_folder(folder_path, double_scaler=False):
    info = pd.read_excel(folder_path+'info.xls')
    output_size = info['output_size'][0]
    block_size = info['block_size'][0]
    network = initialize_network(input_size=3, output_size=output_size,
                                 trained_model=folder_path+'network',
                                 block_size=block_size)

    is_output_relative = info['is_output_relative'][0]
    if is_output_relative or double_scaler:
        scaler_inp = load(folder_path + 'pendulum_scaler_inp.joblib')
        scaler_lab = load(folder_path + 'pendulum_scaler_lab.joblib')
        scaler = [scaler_inp, scaler_lab]
    else:
        scaler = load(folder_path + 'pendulum_scaler.joblib')

    return network, scaler, info


def simple_train(train_data_in, train_data_out, epochs=500):
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

    network, loss_val = pendulum_train_explicit_1step(train_data_in, train_data_out, optimizer=optimizer,
                                                      network=network, epochs=epochs, scaler=scaler)

    return network, scaler