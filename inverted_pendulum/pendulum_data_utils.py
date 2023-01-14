import warnings

import numpy as np
import os
from sys import stdout
import sklearn.preprocessing as scalers
import torch.utils.data as udata
from copy import copy


def pendulum_dataset(filepath, verbose=False):
    """
    The final dataset will contain all the sub-datasets collected. Given the
    directory they are in, all txt files (forces and positions for now) will
    be located and executed.
    """
    # Get a list of the various txt files
    list_of_files = []

    monster_states = np.empty((1, 2))
    monster_inputs = np.empty((1, 1))

    state_data = {}
    input_data = {}

    for (dirpath, dirnames, filenames) in os.walk(filepath):
        filenames.sort()
        for txtname in filenames:
            if txtname.startswith('dataset'):
                list_of_files.append(os.path.join(dirpath, txtname))

    print("Retrieved %d files from base directory '%s' " % (len(list_of_files), filepath))
    if verbose:
        for i in range(len(list_of_files)):
            print("%d %s  \n" % (i, list_of_files[i]))

    for file_idx, filename in enumerate(list_of_files):
        x1, x2, u = np.loadtxt(filename, usecols=(0, 1, 2), skiprows=0, delimiter=',', unpack=True)
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        u = u.reshape(-1, 1)
        states = np.concatenate((x1, x2), axis=1)
        input_data.update({file_idx: u})
        state_data.update({file_idx: states})
        monster_states = np.append(monster_states, states, axis=0)
        monster_inputs = np.append(monster_inputs, u, axis=0)

    monster_states = monster_states[1:, ]  # remove the first "trash" line that was created with np.empty
    monster_inputs = monster_inputs[1:, ]

    monster = np.concatenate((monster_states, monster_inputs), axis=1)

    return state_data, input_data, monster


def reset_pendulum_dataset(state_data, input_data):
    combined_dict = {}
    for dataset_id in state_data.keys():
        combined_dict.update({dataset_id: np.concatenate(
            (state_data[dataset_id], input_data[dataset_id]), axis=1)})
    return combined_dict


def get_block(data, idx, block_size, inp_start_ind=None, input_from_future=True):
    from_idx = idx * block_size
    to_idx = (idx + 1) * block_size - 1
    if inp_start_ind is not None and input_from_future:
        from_idx_inp = from_idx + block_size - 1
        to_idx_inp = to_idx + block_size - 1
    else:
        from_idx_inp = from_idx
        to_idx_inp = to_idx

    success = 1
    # check if idx is out of bounds
    if to_idx_inp > data.shape[0]:
        # print("Not enough points to make a block, dropping it")
        warnings.warn("Not enough points to make a block. Check for the number of the beast.")
        success = 0
    if success:
        block = copy(data[from_idx:to_idx + 1, :])
        if inp_start_ind is not None:
            block[:, inp_start_ind:] = data[from_idx_inp:to_idx_inp + 1, inp_start_ind:]
    else:
        # Just a number to make debugging easier
        block = 666 * np.ones((block_size, data.shape[1]))
    return block


def get_relative_state(current_block, previous_block, output_size=2):
    """
    Function to return a block of the same size but with relative state based on the last step of the previous block.
    @param current_block: the block to be transformed
    @param previous_block: the block used as a basis for the relative state
    @param output_size: refers to the output size of the predictive network == the size of the state considered
    @return relative: the relative state block that is equivalent to current_block
    """
    relative = np.copy(current_block)
    relative[:, :output_size] = relative[:, :output_size] - previous_block[-1, :output_size]
    return relative
