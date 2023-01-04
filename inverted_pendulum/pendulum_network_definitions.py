import torch
import torch.nn as nn
from scipy.stats import norm


class PendulumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers=1, output_size=2, batch_size=1):
        super(PendulumLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.activation = nn.Tanh()

        self.fc_in = nn.Linear(self.input_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc_out = nn.Linear(self.hidden_size * self.seq_len, self.output_size * self.seq_len)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, inputs, hidden):
        inputs_lin = self.activation(self.fc_in(inputs))
        inputs_lin = inputs_lin.view(self.seq_len, self.batch_size, -1)
        lstm_out, hidden = self.lstm(inputs_lin, hidden)
        out = self.fc_out(lstm_out.view(self.seq_len * self.hidden_size))
        return out, hidden


# class PendulumFC(nn.Module):
#     def __init__(self, input_size, seq_len, output_size, batch_size=1):
#         super(PendulumFC, self).__init__()
#         self.activation = nn.ReLU()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#
#         self.fc_in = nn.Linear(self.batch_size * self.input_size * self.seq_len,
#                                self.batch_size * self.input_size * self.seq_len)
#         self.fc2 = nn.Linear(self.batch_size * self.input_size * self.seq_len,
#                              self.batch_size * self.input_size * self.seq_len)
#         # self.fc3 = nn.Linear(self.batch_size*self.input_size * self.seq_len, self.batch_size*self.input_size * self.seq_len)
#         self.fc_out = nn.Linear(self.batch_size * self.input_size * self.seq_len,
#                                 self.batch_size * self.output_size * self.seq_len)
#
#     def forward(self, x):
#         x = torch.flatten(x)
#         output = self.fc_in(                x)
#         output = self.activation(output)
#         output = self.fc2(output)
#         # output = self.activation(output)
#         # output = self.fc3(output)
#         # output = self.activation(output)
#         output = self.fc_out(output)
#         return output

class PendulumFC(nn.Module):
    def __init__(self, input_size, seq_len, output_size, batch_size=1):
        super(PendulumFC, self).__init__()
        self.activation = nn.Tanh()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.fc_in = nn.Linear(self.batch_size * self.input_size * self.seq_len, 50)
        self.fc_h1h2 = nn.Linear(50, 80)
        self.fc_h2h3 = nn.Linear(80, 100)
        self.fc_h3h4 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, self.batch_size * self.output_size * self.seq_len)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.fc_h1h2(output)
        output = self.activation(output)
        output = self.fc_h2h3(output)
        output = self.activation(output)
        output = self.fc_h3h4(output)
        output = self.activation(output)
        output = self.fc_out(output)
        return output


class PendulumEncoder(nn.Module):
    def __init__(self):
        super(PendulumEncoder, self).__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(in_features=40, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), -1)
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class PendulumDecoder(nn.Module):
    def __init__(self):
        super(PendulumDecoder, self).__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(in_features=2, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=40)

    def forward(self, encoding):
        x = self.activation(self.fc1(encoding))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class PendulumAutoEncoder(nn.Module):
    def __init__(self, PendulumEncoder, PendulumDecoder):
        super(PendulumAutoEncoder, self).__init__()
        self.encoder = PendulumEncoder
        self.decoder = PendulumDecoder

    def forward(self, current_block, next_block):
        encode = self.encoder(current_block, next_block)
        decode = self.decoder(encode)
        return decode


class PendulumEncoderMultistep(nn.Module):
    def __init__(self, prediction_horizon, block_length):
        super(PendulumEncoderMultistep, self).__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length,
                             out_features=2 * (prediction_horizon + 1) * block_length - 10)
        self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 10,
                             out_features=2 * (prediction_horizon + 1) * block_length - 20)
        self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 20,
                             out_features=2 * (prediction_horizon + 1) * block_length - 30)
        self.fcout = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 30,
                               out_features=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fcout(x)
        return x


class PendulumDecoderMultistep(nn.Module):
    def __init__(self, prediction_horizon, block_length):
        super(PendulumDecoderMultistep, self).__init__()
        self.activation = nn.Tanh()
        self.fcin = nn.Linear(in_features=2,
                              out_features=2 * (prediction_horizon + 1) * block_length - 30)
        self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 30,
                             out_features=2 * (prediction_horizon + 1) * block_length - 20)
        self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 20,
                             out_features=2 * (prediction_horizon + 1) * block_length - 10)
        self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 10,
                             out_features=2 * (prediction_horizon + 1) * block_length)

    def forward(self, encoding):
        x = self.activation(self.fcin(encoding))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class PendulumAutoEncoderMultistep(nn.Module):
    def __init__(self, PendulumEncoderMultistep, PendulumDecoderMultistep):
        super(PendulumAutoEncoderMultistep, self).__init__()
        self.encoder = PendulumEncoderMultistep
        self.decoder = PendulumDecoderMultistep

    def forward(self, current_block, next_block):
        encode = self.encoder(current_block, next_block)
        decode = self.decoder(encode)
        return decode

class PendulumVariationalEncoderMultistep(nn.Module):
    def __init__(self, prediction_horizon, block_length):
        super(PendulumVariationalEncoderMultistep, self).__init__()
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length,
                             out_features=2 * (prediction_horizon + 1) * block_length - 4)
        self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 4,
                             out_features=2 * (prediction_horizon + 1) * block_length - 12)
        self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 12,
                             out_features=2 * (prediction_horizon + 1) * block_length - 17)
        self.fcout_mu = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 17,
                                  out_features=2)
        self.fcout_sigma = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 17,
                                     out_features=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x_mu = self.fcout_mu(x)
        x_logvar = self.fcout_sigma(x)
        return x_mu, x_logvar


class PendulumVariationalDecoderMultistep(nn.Module):
    def __init__(self, prediction_horizon, block_length):
        super(PendulumVariationalDecoderMultistep, self).__init__()
        self.activation = nn.Tanh()
        self.fcin = nn.Linear(in_features=2,
                              out_features=2 * (prediction_horizon + 1) * block_length - 17)
        self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 17,
                             out_features=2 * (prediction_horizon + 1) * block_length - 12)
        self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 12,
                             out_features=2 * (prediction_horizon + 1) * block_length - 4)
        self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 4,
                             out_features=2 * (prediction_horizon + 1) * block_length)

    def forward(self, encoding):
        x = self.activation(self.fcin(encoding))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class PendulumVariationalAutoEncoderMultistep(nn.Module):
    def __init__(self, PendulumVariationalEncoderMultistep, PendulumVariationalDecoderMultistep):
        super(PendulumVariationalAutoEncoderMultistep, self).__init__()
        self.encoder = PendulumVariationalEncoderMultistep
        self.decoder = PendulumVariationalDecoderMultistep

    def forward(self, current_block, next_block):
        mu, logvar = self.encoder(current_block, next_block)
        z = self.reparametrize(mu, logvar)
        decode = self.decoder(z)
        return decode, mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)



# class PendulumVariationalEncoderMultistep(nn.Module):
#     def __init__(self, prediction_horizon, block_length):
#         super(PendulumVariationalEncoderMultistep, self).__init__()
#         self.activation = nn.Tanh()
#         self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length,
#                              out_features=2 * (prediction_horizon + 1) * block_length - 10)
#         self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 10,
#                              out_features=2 * (prediction_horizon + 1) * block_length - 20)
#         self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 20,
#                              out_features=2 * (prediction_horizon + 1) * block_length - 30)
#         self.fcout_mu = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 30,
#                                   out_features=2)
#         self.fcout_sigma = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 30,
#                                      out_features=2)
#
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2))
#         x = x.flatten()
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
#         x_mu = self.fcout_mu(x)
#         x_logvar = self.fcout_sigma(x)
#         return x_mu, x_logvar
#
#
# class PendulumVariationalDecoderMultistep(nn.Module):
#     def __init__(self, prediction_horizon, block_length):
#         super(PendulumVariationalDecoderMultistep, self).__init__()
#         self.activation = nn.Tanh()
#         self.fcin = nn.Linear(in_features=2,
#                               out_features=2 * (prediction_horizon + 1) * block_length - 30)
#         self.fc1 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 30,
#                              out_features=2 * (prediction_horizon + 1) * block_length - 20)
#         self.fc2 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 20,
#                              out_features=2 * (prediction_horizon + 1) * block_length - 10)
#         self.fc3 = nn.Linear(in_features=2 * (prediction_horizon + 1) * block_length - 10,
#                              out_features=2 * (prediction_horizon + 1) * block_length)
#
#     def forward(self, encoding):
#         x = self.activation(self.fcin(encoding))
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# class PendulumVariationalAutoEncoderMultistep(nn.Module):
#     def __init__(self, PendulumVariationalEncoderMultistep, PendulumVariationalDecoderMultistep):
#         super(PendulumVariationalAutoEncoderMultistep, self).__init__()
#         self.encoder = PendulumVariationalEncoderMultistep
#         self.decoder = PendulumVariationalDecoderMultistep
#
#     def forward(self, current_block, next_block):
#         mu, logvar = self.encoder(current_block, next_block)
#         z = self.reparametrize(mu, logvar)
#         decode = self.decoder(z)
#         return decode, mu, logvar
#
#     def reparametrize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
