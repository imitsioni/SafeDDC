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


'''
Returns a *dataset* where the absolute positions at X,Y,Z have been replaced by relative displacements.
'''


def get_relative_dataset(dataset, block_size, verbose=0):
    previous_block = get_block(dataset, 0, block_size)
    relative_dataset = np.empty((0, dataset.shape[1]))
    tot_blocks = int(np.floor(dataset.shape[0] / block_size))

    for i in range(1, tot_blocks):
        if verbose:
            stdout.write("\r%d out of %d" % (i, tot_blocks))
            stdout.flush()
        current_block = get_block(dataset, i, block_size)
        relative_dataset = np.append(relative_dataset, get_relative_state(current_block, previous_block), axis=0)
        previous_block = get_block(dataset, i, block_size)

    return relative_dataset


class TXTPendulumImporter(object):
    """
    Data handler for importing the training data gathered on the robot.
    Inputs:
        base_path: Path to the training data
        data_type: The type of data to be loaded, "train" or "test"
        block_size: Length M of the sequence comprising each block. 10 in implementation.
        block_horizon: Prediction horizon
        scaler: scaler used for input data, if already computed during training
    """

    def __init__(self, base_path, data_type, block_size, block_horizon, scale_data, skips=1,
                 scaler=None):
        self.data = []
        self.base_path = base_path
        self.block_size = block_size
        self.data_type = data_type
        self.block_horizon = block_horizon
        # Initialization
        self.state_data = {}
        self.input_data = {}
        self.combined_datasets = {}
        self.idxToRunDict = {}
        self.skips = skips

        # Read the data
        self.importDataset()
        self.scale_data = scale_data

        if self.data_type == "train":
            # calculate scalers if running through training data
            self.scaler = scalers.StandardScaler()
            self.getScalerParams()
        else:
            # otherwise receive previously computed scalers
            self.scaler = scaler
            relative_full = self.createFull()

        self.createIdxToRunBlockDict()

    def importDataset(self, verbose=False):
        """
        The full dataset contains all the sub-datasets for each object and setting. Given the
        directory they are in, all txt files will be located and incorporated.
        """
        # Get a list of the various txt files
        list_of_files = []

        monster_states = np.empty((1, 2))
        monster_inputs = np.empty((1, 1))

        state_data = {}
        input_data = {}

        for (dirpath, dirnames, filenames) in os.walk(self.base_path):
            dirnames.sort()
            for txtname in filenames:
                if txtname.startswith('dataset'):
                    list_of_files.append(os.path.join(dirpath, txtname))

        print("Retrieved %d files from base directory '%s' " % (len(list_of_files), self.base_path))

        for file_idx, filename in enumerate(list_of_files):
            x1, x2, u = np.loadtxt(filename, usecols=(0, 1, 2), skiprows=0, delimiter=',', unpack=True)
            x1 = x1.reshape(-1, 1)
            x2 = x2.reshape(-1, 1)
            u = u.reshape(-1, 1)
            states = np.concatenate((x1, x2), axis=1)
            self.input_data.update({file_idx: u})
            self.state_data.update({file_idx: states})
            monster_states = np.append(monster_states, states, axis=0)
            monster_inputs = np.append(monster_inputs, u, axis=0)

        monster_states = monster_states[1:, ]  # remove the first "trash" line that was created with np.empty
        monster_inputs = monster_inputs[1:, ]

        monster = np.concatenate((monster_states, monster_inputs), axis=1)

        return state_data, input_data, monster

    def createFull(self):
        """
        Creates the full dataset that we use to fit the normalizer and also
        builds a dictionary of {d_idx, dataset} for future use.

        """
        combined_dict = reset_pendulum_dataset(self.state_data, self.input_data)
        full_dataset = np.empty((1, 3))
        for d_idx, dataset in combined_dict.items():
            self.combined_datasets.update({d_idx: dataset})
            full_dataset = np.append(full_dataset, dataset, axis=0)
        return full_dataset[1:, :]

    def getScalerParams(self):
        """
        Calculates the scaler required for full input dataset
        """
        # first get the relative_full dataset
        relative_full = self.createFull()
        self.scaler.fit(relative_full)
        if self.scale_data:
            print("self.scaler.mean_", self.scaler.mean_)
        else:
            print("Data will not be scaled!")

    def createIdxToRunBlockDict(self):
        """
        Creates a dictionary that translates between full dataset index, and sub-dataset ID + block id
        to be fetched from the TXT importers.
        """
        #todo STRANGER DANGER, THE +1 IS BECAUSE WE'RE NOT DOING RELATIVE
        dictLengths = [int(len(self.combined_datasets[dk]) / self.block_size) - self.block_horizon for dk in
                       self.combined_datasets.keys()]
        totalLength = np.sum(dictLengths)

        currDictIdx = 0
        accrued = 0
        for i in range(totalLength):
            if i >= dictLengths[currDictIdx] + accrued:
                accrued += dictLengths[currDictIdx]
                currDictIdx += 1
            self.idxToRunDict.update({i: (currDictIdx, i - accrued)})
        print("len(idxToRunDict)", len(self.idxToRunDict))

    def getBlock(self, block_idx):
        """
        Returns the input block and label for a given dataset index
        """
        d_idx, b_idx = self.idxToRunDict[block_idx]
        current_block_frame = get_block(self.combined_datasets[d_idx], b_idx, self.block_size)
        block_label = self.getLabel(block_idx)
        if self.scale_data:
            current_block_frame_scaled = self.scaler.transform(current_block_frame)
            return current_block_frame_scaled, block_label
        else:
            return current_block_frame, block_label

    def getLabel(self, block_idx):
        """
        Returns the SCALED state of the block 'self.block_horizon' ahead in the future
        """
        d_idx, b_idx = self.idxToRunDict[block_idx]
        next_block_frame = get_block(self.combined_datasets[d_idx], b_idx + self.block_horizon, self.block_size)
        next_block_frame_scaled = self.scaler.transform(next_block_frame)
        if self.scale_data:
            return next_block_frame_scaled[:, :2]
        else:
            return next_block_frame[:, :2]


class PendulumDataloader(udata.Dataset):
    """
    Underlying dataset handler for the data.
    Arguments:
        base_path: Path to data
        data_type: The type of data to be loaded, "train" or "test"
        block_size: Length M of the sequence comprising each block. 10 in implementation.
        block_horizon: Prediction horizon
        scaler: scaler used for input data, if already computed during training.
    ------------------------------------
    Usage example:
    ------------------------------------
    data_path = '*path to datasets*/inv_damp_data/2D'

    horizon = 3
    block_size = 10
    batch_size = 8
    # Initialize the underlying class
    train_dl = PendulumDataloader(base_path=data_path, data_type='train', block_size=block_size, block_horizon=horizon)
    # Initialize the dataloader
    train_loader = torch.utils.data.DataLoader(train_dl, batch_size, shuffle=False, drop_last=True)
    for idx, data in enumerate(train_loader):
        inputs, labels = data
    """

    def __init__(self, base_path, data_type, block_size, block_horizon, scale_data, skips=1,
                 scaler=None):
        self.dh = TXTPendulumImporter(base_path=base_path, data_type=data_type, block_size=block_size,
                                      block_horizon=block_horizon, scale_data=scale_data, skips=skips, scaler=scaler)

    def __len__(self):
        return len(self.dh.idxToRunDict)

    def __getitem__(self, idx):
        return self.dh.getBlock(idx)
