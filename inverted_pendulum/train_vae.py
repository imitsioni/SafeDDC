import torch
from sklearn import preprocessing
import matplotlib.pyplot as plt

import numpy as np
from joblib import dump

from inverted_pendulum import pendulum_data_utils as du
from inverted_pendulum import pendulum_network_definitions as nd
from inverted_pendulum import pendulum_train_utils as tu
from pendulum_network_interface import load_network_from_folder


seed = 254517766290206988366
torch.manual_seed(seed)


train_data_path = 'rrtc+random/converged_trials'
is_output_relative = True

state_data_train, input_data_train, monster_train = du.pendulum_dataset(train_data_path)
data_len = len(state_data_train)
train_rate = 0.9
n_train = np.floor(data_len * train_rate)
train_data_in = np.zeros((0, 3))
train_data_out = np.zeros((0, 2))
valid_data_in = np.zeros((0, 3))
valid_data_out = np.zeros((0, 2))
for i in range(len(state_data_train)):
    new_data_in = np.concatenate((state_data_train[i][:-1, :3], input_data_train[i][:-1, :3]), axis=1)
    if is_output_relative:
        new_data_out = state_data_train[i][1:, :2] - state_data_train[i][:-1, :2]
    else:
        new_data_out = state_data_train[i][1:, :2]
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
dump(scaler_in, 'rrtc+random_converged_scaler.joblib')
dump(scaler_out, 'rrtc+random_converged_delta_scaler.joblib')


# -------------- Training the AE
train_dict = du.reset_pendulum_dataset(state_data_train, input_data_train)

output_size = 2
block_size = 1
final_horizon = 10
epochs = 25

encoder = nd.PendulumVariationalEncoderMultistep(final_horizon, block_size)
decoder = nd.PendulumVariationalDecoderMultistep(final_horizon, block_size)
model = nd.PendulumVariationalAutoEncoderMultistep(encoder, decoder)

lr = 1e-3
wd = 1e-4
beta = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


model, training_loss, validation_loss = tu.pendulum_train_multistep_AE(train_dict, None, optimizer,
                                model, epochs, scaler_in, final_horizon, output_size, block_size, beta, vae=True)
plt.figure()
plt.plot(training_loss, label='training')
plt.legend()
plt.show()

model_name = "trained_models/vae_lr" + str(lr) + "_wd_" + str(wd) + "_bet_" + str(beta) + "_hor_" \
             + str(final_horizon) + "_epo_" + str(epochs) + str(seed)

torch.save(model.state_dict(), model_name+".pt")

