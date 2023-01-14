import numpy as np
import torch
import torch.nn.functional as F
from inverted_pendulum import pendulum_data_utils as du
import logging


def vae_loss_function(recon_x, x, mu, logvar, beta):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta*KLD, MSE, beta*KLD


def custom_mse(prediction, target):
    loss = (prediction - target) ** 2
    ls = loss.sum()
    ls = ls / len(target)
    return ls


def pendulum_train_multistep_AE(training_dataset_dict, validation_dataset_dict, optimizer,
                                model, epochs, scaler, final_horizon, output_size, block_size, beta, vae=True):
    train_losses = []
    validation_losses = []
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        # Losses per dataset in training_dict
        dataset_losses = []
        mse_losses = []
        kld_losses = []
        for d_idx, dataset in training_dataset_dict.items():
            tot_blocks = int(np.floor(dataset.shape[0]) / block_size)
            # Losses for the group of blocks we're iterating over
            block_losses = []
            block_mse_losses = []
            block_kld_losses = []
            for block in range(tot_blocks - final_horizon - 1):
                # Get the current block and scale it
                current_in = du.get_block(dataset, block, block_size, output_size, False)
                current_scaled = scaler.transform(current_in)
                sequence_buffer = []
                for bl_ind in range(1, final_horizon + 1):
                    next_block = du.get_block(dataset, block + bl_ind, block_size, output_size, False)  # absolute and unscaled
                    next_block = scaler.transform(next_block)  # scaled
                    sequence_buffer.append(next_block[:, :output_size])
                optimizer.zero_grad()
                current_in_tensor = torch.tensor(current_scaled[None, :, :output_size]).float()
                next_blocks_tensor = torch.tensor(sequence_buffer).float()
                groundtruth = torch.cat((current_in_tensor, next_blocks_tensor)).flatten()
                if vae:
                    out, mu, logvar = model(current_in_tensor, next_blocks_tensor)
                    reconstruction_loss, mse, kld = vae_loss_function(out, groundtruth, mu, logvar, beta)
                else:
                    out = model(current_in_tensor, next_blocks_tensor)
                    reconstruction_loss = criterion(groundtruth, out)
                block_losses.append(reconstruction_loss.detach().numpy())
                block_mse_losses.append(mse.detach().numpy())
                block_kld_losses.append(kld.detach().numpy())

                reconstruction_loss.backward()
                optimizer.step()
            dataset_losses.append(np.mean(block_losses))
            mse_losses.append(np.mean(block_mse_losses))
            kld_losses.append(np.mean(block_kld_losses))

        epoch_loss = np.mean(dataset_losses)
        train_losses.append(epoch_loss)
        # Going for validation
        # validation_loss = pendulum_validate_multistep_AE(validation_dataset_dict, model, scaler,
        #                                                  final_horizon, output_size, block_size, beta, vae)
        # validation_losses.append(validation_loss)
        validation_loss = 0
        print(epoch, epoch_loss, validation_loss)
        print("-- MSE:", np.mean(mse_losses), '--KLD:', np.mean(kld_losses))
    return model, train_losses, validation_losses


def pendulum_validate_multistep_AE(validation_dataset_dict, model, scaler, final_horizon, output_size, block_size, beta, vae):
    dataset_losses = []
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        model.eval()
        for d_idx, dataset in validation_dataset_dict.items():
            tot_blocks = int(np.floor(dataset.shape[0]) / block_size)
            # Losses for the group of blocks we're iterating over
            block_losses = []
            for block in range(tot_blocks - final_horizon - 1):
                # Get the current block and scale it
                current_in = du.get_block(dataset, block, block_size, output_size, False)
                current_scaled = scaler.transform(current_in)
                sequence_buffer = []
                for bl_ind in range(1, final_horizon + 1):
                    next_block = du.get_block(dataset, block + bl_ind, block_size, output_size, False)  # absolute and unscaled
                    next_block = scaler.transform(next_block)  # scaled
                    sequence_buffer.append(next_block[:, :output_size])
                current_in_tensor = torch.tensor(current_scaled[None, :, :output_size]).float()
                next_blocks_tensor = torch.tensor(sequence_buffer).float()
                groundtruth = torch.cat((current_in_tensor, next_blocks_tensor)).flatten()
                if vae:
                    out, mu, logvar = model(current_in_tensor, next_blocks_tensor)
                    reconstruction_loss, mse, kld = vae_loss_function(out, groundtruth, mu, logvar, beta)
                else:
                    out = model(current_in_tensor, next_blocks_tensor)
                    reconstruction_loss = criterion(groundtruth, out)
                block_losses.append(reconstruction_loss.detach().numpy())
            dataset_losses.append(np.mean(block_losses))
    validation_loss = np.mean(dataset_losses)
    return validation_loss


def pendulum_train_explicit_1step(data_in_train, data_out_train, optimizer, network, epochs, scaler):
    loss_val = []
    in_tensor_unscaled = torch.tensor(data_in_train).float()
    out_tensor_unscaled = torch.tensor(data_out_train).float()
    if isinstance(scaler, list):
        in_tensor_scaled = torch.tensor(scaler[0].transform(in_tensor_unscaled)).float()
        out_tensor_scaled = torch.tensor(scaler[1].transform(out_tensor_unscaled)).float()
    else:
        in_tensor_scaled = torch.tensor(scaler.transform(in_tensor_unscaled)).float()
        out_tensor_scaled = torch.tensor(scaler.transform(out_tensor_unscaled)).float()

    for epoch in range(epochs):

        optimizer.zero_grad()
        out_pred = network(in_tensor_scaled)
        loss = F.mse_loss(out_pred, out_tensor_scaled)
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())
        loss_val.append(loss.item())

        # if epoch % 10 == 0:
        #     print('\t Validation round')
        #     validation_losses.append(
        #         val_pendulum_FC_single_timestep(validation_dataset_dict, network, scaler, block_size, output_size))

    return network, loss_val

    in_tensor_unscaled = torch.tensor(data_in_train).float()
    out_tensor_unscaled = torch.tensor(data_out_train).float()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out_pred = network(in_tensor_unscaled)
        loss = F.mse_loss(out_pred, out_tensor_unscaled)
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())

    return network