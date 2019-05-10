from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import time
from multiprocessing import Process, Queue, Pipe
import copy
from pprint import pprint
import sys

class ReplayBuffer(object):
    """
    Replay buffer meant to work in a synchrounous environment
    """
    def __init__(self, size=20e6, state_size, traj_len=100):
        self.size = size #what, are you even going to fill this up?
        self.traj_len = traj_len

        self.obs_buffer = np.zeros((size, traj_len, state_size[0], 
            state_size[1], state_size[2]))
        self.pos_buffer = np.zeros((size, traj_len, 2))
        self.dir_buffer = np.zeros((size, traj_len))
        self.num_elements = 0

    def add(self, obs, pos, dirs):
        assert obs.shape[0] == pos.shape[0]
        assert obs.shape[0] == dirs.shape[0]
        batch_size = obs.shape[0]
        self.obs_buffer[self.num_elements:self.num_elements+batch_size] = obs
        self.pos_buffer[self.num_elements:self.num_elements+batch_size] = pos
        self.dir_buffer[self.num_elements:self.num_elements+batch_size] = dirs
        self.num_elements += batch_size

    def sample(self, sample_size):
        if sample_size < len(self.buffer):
            print("Ah! Should be more elements in buffer first!")
        print("shape of samples:")
        # Wont work with >1D array
        indecies = np.random.choice(len(self.buffer), size=sample_size, 
            replace=True)

        obs = np.array([self.obs_buffer[i] for i in indecies])
        pos = np.array([self.pos_buffer[i] for i in indecies])
        dirs = np.array([self.dir_buffer[i] for i in indecies])

        return obs, pos, dirs



if __name__ == '__main__':
    pass