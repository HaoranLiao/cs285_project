import numpy as np
from numpy.random import RandomState
import random
import pickle

class BatchGenerator():
    def __init__(self, data, batch_size, random_seed=1) -> None:
        self.data = data
        self.batch_size = batch_size
        self.idx_buffer = list(range(len(data)))
        self.random_state = RandomState(random_seed)
        self.random_state.shuffle(self.idx_buffer)

    def next_batch(self):
        if len(self.idx_buffer) < self.batch_size:
            new_idx = list(range(len(self.data)))
            self.random_state.shuffle(new_idx)
            self.idx_buffer += new_idx
        selected_idx = self.idx_buffer[:self.batch_size]
        self.idx_buffer = self.idx_buffer[self.batch_size:]
        batch_x = self.data[selected_idx]
        batch_x = batch_x / 255
        return batch_x

    def get_all_data(self):
        data = self.data[self.idx_buffer]
        data = data / 255
        return data


# def cifar10_next_batch(num, x):
#     """
#     Return a total of `num` samples from x
#     """
#     idx = np.arange(0, len(x))     # get all possible indexes
#     np.random.shuffle(idx)         # shuffle indexes
#     idx = idx[0:num]               # use only `num` random indexes
#     batch_x = [x[i] for i in idx]  # get list of `num` random samples
#     batch_x = np.asarray(batch_x)  # get back numpy array
#     # batch_x = batch_x.reshape((num, 32*32*3))
#     batch_x = batch_x / 255
#     return batch_x

def read_cifar_data(batch_idx_list=None):
    returned_content = []
    for idx in batch_idx_list:
        with open(f"cifar-10-batches-py/data_batch_{idx}", "rb") as f:
            content = pickle.load(f, encoding="bytes")
            returned_content.append(content[b"data"])
    returned_content = np.concatenate(returned_content)
    returned_content = np.moveaxis(returned_content.reshape((-1, 3, 32, 32)), 1, -1)
    return returned_content

def read_cifar_test_data():
    with open("cifar-10-batches-py/test_batch", "rb") as f:
        content = pickle.load(f, encoding="bytes")
        returned_content = content[b"data"]
    returned_content = np.moveaxis(returned_content.reshape((-1, 3, 32, 32)), 1, -1)
    return returned_content

def read_atari_observations():
    observations = []
    for i in range(5):
        with open(f"../data/mspacman/expert_data_{i}.pkl", "rb") as f:
            data = pickle.load(f)
            for traj in data:
                observations.append(traj["observation"])
    observations = np.concatenate(observations)
    return observations