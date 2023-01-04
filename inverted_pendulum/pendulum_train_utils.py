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


def pendulum_train_multistep_curriculum(training_dataset_dict, validation_dataset_dict, optimizer, scheduler, model,
                                        epochs, scalers, final_horizon, output_size=2, block_size=10, horizon_switch=20):
    train_losses = []
    validation_losses = []
    horizon = 1
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        # Logging
        if (epoch + 1) % horizon_switch == 0 and horizon < final_horizon:
            horizon += 1
            print("Epoch: ", epoch, "LR: ", scheduler[0].get_last_lr(), "Horizon: ", horizon)

        # Losses per dataset in training_dict
        dataset_losses = []
        for d_idx, dataset in training_dataset_dict.items():
            tot_blocks = int(np.floor(dataset.shape[0]) / block_size)
            # Losses for the group of blocks we're iterating over
            block_losses = []
            for block in range(tot_blocks - horizon - 1):
                # Get the current block and scale it
                current_in = du.get_block(dataset, block, block_size, inp_start_ind=output_size)
                current_scaled = scalers[0].transform(current_in)
                optimizer.zero_grad()
                # todo: If switching to lstm you need this
                # net_hidden = model.init_hidden()
                # Buffers to keep the predictions
                pred_out = torch.zeros([0])
                real_out = torch.zeros([0])
                iterative_in = torch.tensor(current_scaled).float()
                for bl_ind in range(1, horizon + 1):
                    # out, net_hidden = model(iterative_in, net_hidden)  # this will serve as the new scaled input
                    out = model(iterative_in)  # delta
                    # Update the buffer
                    pred_out = torch.cat((pred_out, out))
                    # Fetch the next block
                    next_block = du.get_block(dataset, block + bl_ind, block_size, inp_start_ind=output_size,
                                              input_from_future=True)  # absolute and unscaled
                    tgt = du.get_relative_state(next_block, current_in)  # delta unscaled
                    delta_scaled = scalers[1].transform(tgt[:, :output_size])  # delta scaled
                    next_block_tensor = torch.tensor(delta_scaled).float()
                    # add them to the labels placeholder
                    real_out = torch.cat((real_out, next_block_tensor.flatten()))

                    next_block_scaled = scalers[0].transform(next_block)  # full block (not delta) scaled
                    # we unfortunately need to switch back to unscaled for the new inputs
                    unscaled_delta = scalers[1].inverse_transform(out.detach().view(block_size, output_size))

                    # figure out the unscaled, estimated next input based on the delta
                    next_pred = unscaled_delta + current_in[-1, :output_size]  # unscaled and absolute
                    next_full = torch.cat((torch.tensor(next_pred),
                                           torch.tensor(next_block[:, output_size:])), 1)  # the full deal, unscaled

                    iterative_in = torch.tensor(scalers[0].transform(next_full)).float()
                    current_in = next_block
                # loss = criterion(pred_out, real_out)
                loss = custom_mse(pred_out.view((horizon + 1) * block_size, -1),
                                  real_out.view((horizon + 1) * block_size, -1))

                block_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            dataset_losses.append(np.mean(block_losses))
        epoch_loss = np.mean(dataset_losses)
        train_losses.append(epoch_loss)
        # Going for validation
        validation_loss = pendulum_validate_multistep_curriculum(validation_dataset_dict, model, scalers, final_horizon,
                                                                 output_size, block_size)
        validation_losses.append(validation_loss)
        scheduler[0].step()
        scheduler[1].step(validation_loss)
        print(epoch, epoch_loss, validation_loss)
    return model, train_losses, validation_losses


def pendulum_validate_multistep_curriculum(validation_dataset_dict, model, scalers, horizon, output_size, block_size):
    dataset_losses = []
    # criterion = torch.nn.MSELoss()
    criterion = custom_mse
    model.eval()
    with torch.no_grad():
        for d_idx, dataset in validation_dataset_dict.items():
            tot_blocks = int(np.floor(dataset.shape[0]) / block_size)
            # Losses for the group of blocks we're iterating over
            block_losses = []
            for block in range(tot_blocks - horizon - 1):
                # Get the current block and scale it
                current_in = du.get_block(dataset, block, block_size, inp_start_ind=output_size)
                current_scaled = scalers[0].transform(current_in)
                # todo: If switching to lstm you need this
                # net_hidden = model.init_hidden()
                # Buffers to keep the predictions
                pred_out = torch.zeros([0])
                real_out = torch.zeros([0])
                iterative_in = torch.tensor(current_scaled).float()
                for bl_ind in range(1, horizon + 1):
                    out = model(iterative_in)  # delta
                    # Update the buffer
                    pred_out = torch.cat((pred_out, out))
                    # Fetch the next block
                    next_block = du.get_block(dataset, block + bl_ind, block_size, inp_start_ind=output_size,
                                              input_from_future=True)  # absolute and unscaled
                    tgt = du.get_relative_state(next_block, current_in)  # delta unscaled
                    delta_scaled = scalers[1].transform(tgt[:, :output_size])  # delta scaledd
                    next_block_tensor = torch.tensor(delta_scaled).float()
                    # add them to the labels placeholder
                    real_out = torch.cat((real_out, next_block_tensor.flatten()))

                    next_block_scaled = scalers[0].transform(next_block)  # full block (not delta) scaled
                    # we unfortunately need to switch back to unscaled for the new inputs
                    unscaled_delta = scalers[1].inverse_transform(out.detach().view(block_size, output_size))

                    # figure out the unscaled, estimated next input based on the delta
                    next_pred = unscaled_delta + current_in[-1, :output_size]  # unscaled and absolute
                    next_full = torch.cat((torch.tensor(next_pred),
                                           torch.tensor(next_block[:, output_size:])), 1)  # the full deal, unscaled
                    iterative_in = torch.tensor(scalers[0].transform(next_full)).float()
                    current_in = next_block
                loss = criterion(pred_out, real_out)
                block_losses.append(loss.item())
            dataset_losses.append(np.mean(block_losses))
    validation_loss = np.mean(dataset_losses)
    # print("\t  Mean %d-step prediction error is %.6f:  " % (horizon, validation_loss))
    return validation_loss


def pendulum_train_batches(train_loader, val_loader, optimizer, model, epochs):
    """
    Training of a dynamics model using dataloaders
    @param train_loader: Dataloader object with training sets
    @param val_loader: Dataloader object with validation sets
    @param optimizer: Optimizer for training
    @param model: Model to be trained
    @param epochs: Total number of epochs
    @return: the trained model, list of training losses, list of validation losses
    """
    # Train the model
    train_losses = []
    val_losses = []
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        epoch_losses = []
        model = model.train()
        for idx, data in enumerate(train_loader):
            current_state, next_state = data
            net_pred = model(current_state.float())
            optimizer.zero_grad()
            loss = criterion(net_pred.flatten(), next_state.float().flatten())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().numpy())

        epoch_val_losses = []
        model = model.eval()
        for idx, data in enumerate(val_loader):
            current_state, next_state = data
            net_pred = model(current_state.float())
            loss = criterion(net_pred.flatten(), next_state.float().flatten())
            epoch_val_losses.append(loss.detach().numpy())
        if epoch % 10 == 0:
            print("--- Epoch: %d \t Train loss: %f \t Val loss: %f " %
                  (epoch, np.mean(epoch_losses), np.mean(epoch_val_losses)))
        train_losses.append(np.mean(epoch_losses))
        val_losses.append(np.mean(epoch_val_losses))
        # print(len(train_losses), len(val_losses), np.mean(train_losses), np.mean(val_losses))
    print("---- Average train loss: %f \t Validation loss: %f." % (np.mean(train_losses), np.mean(val_losses)))
    return model, train_losses, val_losses


def pendulum_train_FC_single_timestep(training_dataset_dict, validation_dataset_dict, optimizer, network, epochs,
                                      scaler,
                                      output_size):
    """
    Training a dynamics model for one step ahead predictions using TIMESTEPS (not sequences)
    Note that it does not use dataloaders
    @param training_dataset_dict: The training data dictionary
    @param validation_dataset_dict: The validation data dictionary
    @param optimizer: The optimizer used for training
    @param network: Network to be trained
    @param epochs: Total amount of inputs
    @param scaler: Scaler for data pre-processing
    @param output_size: Desired output size (full state or only positions: 1/2)
    @return:
    """
    train_losses = []
    validation_losses = []
    block_size = 1
    for epoch in range(epochs):
        dataset_losses = []
        for d_idx, dataset in training_dataset_dict.items():
            network = network.train()
            block_losses = []
            for step in range(len(dataset) - 1):
                current_in = dataset[step]
                current_in = scaler.transform(current_in.reshape(1, 3))  # 3
                current_tensor_in = torch.tensor(current_in).float()
                current_label = dataset[step + 1]
                current_label_sc = scaler.transform(current_label.reshape(1, 3))
                current_label_tensor = torch.tensor(current_label_sc).float()
                optimizer.zero_grad()
                out = network(current_tensor_in)
                loss = F.mse_loss(out.view(block_size, output_size), current_label_tensor[:, :output_size])
                block_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            d_loss = sum(block_losses) / len(block_losses)
            dataset_losses.append(d_loss)
        epoch_loss = sum(dataset_losses) / len(dataset_losses)
        train_losses.append(epoch_loss)
        print(epoch, epoch_loss)

        if epoch % 10 == 0:
            print('\t Validation round')
            validation_losses.append(
                val_pendulum_FC_single_timestep(validation_dataset_dict, network, scaler, block_size, output_size))

    return train_losses, validation_losses


def val_pendulum_FC_single_timestep(validation_dataset_dict, network, scaler, block_size, output_size):
    validation_losses = []
    dataset_losses = []
    network = network.eval()
    with torch.no_grad():
        for d_idx, dataset in validation_dataset_dict.items():
            network = network.train()
            block_losses = []
            for step in range(len(dataset) - 1):
                current_in = dataset[step]
                current_in = scaler.transform(current_in.reshape(1, 3))  # 3
                current_tensor_in = torch.tensor(current_in).float()
                current_label = dataset[step + 1]
                current_label_sc = scaler.transform(current_label.reshape(1, 3))
                current_label_tensor = torch.tensor(current_label_sc).float()
                out = network(current_tensor_in)
                loss = F.mse_loss(out.view(block_size, output_size), current_label_tensor[:, :output_size])
                block_losses.append(loss.item())
            d_loss = sum(block_losses) / len(block_losses)
            dataset_losses.append(d_loss)
    #                 print("\t \t %d, %f" %(d_idx, d_loss))
    validation_loss = sum(dataset_losses) / len(dataset_losses)
    print("\t Mean 1-step prediction error is : %0.4f " % (validation_loss))
    return validation_loss


def pendulum_train_flow_transitions(train_loader, val_loader, model, optimizer, iterations, model_name=None):
    model = model.train()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    training_loss = []
    validation_loss = []
    for i in range(iterations + 1):
        batch_loss = []
        # Run through the whole dataset
        for idx, data in enumerate(train_loader):
            current_state, next_state = data
            # Bringing positions and forces in the right dimensionality
            curr = current_state[:, :, :2]
            curr = torch.flatten(curr, start_dim=0, end_dim=1)
            nxt = torch.flatten(next_state, start_dim=0, end_dim=1)
            # print(curr.shape, nxt.shape)
            x = torch.cat((curr, nxt), dim=1)
            optimizer.zero_grad()
            z, prior_logprob, log_det = model(x.float())
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)
            batch_loss.append(-loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
        training_loss.append(np.mean(batch_loss))
        validation_loss.append(pendulum_validate_flow_transitions(val_loader, model))
        if i % 10 == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Train Logprob: {np.mean(training_loss):.2f}\t" +
                        f"Val Logprob: {np.mean(validation_loss):.2f}\t"
                        )
    return model, training_loss, validation_loss


def pendulum_validate_flow_transitions(val_loader, model):
    model = model.eval()
    running_loss = []
    for idx, data in enumerate(val_loader):
        current_state, next_state = data
        # Bringing positions and forces in the right dimensionality
        curr = current_state[:, :, :2]
        curr = torch.flatten(curr, start_dim=0, end_dim=1)
        nxt = torch.flatten(next_state, start_dim=0, end_dim=1)
        x = torch.cat((curr, nxt), dim=1)
        z, prior_logprob, log_det = model(x.float())
        logprob = prior_logprob + log_det
        loss = -torch.mean(logprob)
        running_loss.append(-loss.detach().cpu().numpy())
    return np.mean(running_loss)


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


def pendulum_train_explicit_1step_noscale(data_in_train, data_out_train, optimizer, network, epochs):
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