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
    Problem: My estimates give that DM used like 8 TB of data for this... shit
    Solution: Saving only the actions and random seed, then replicating the traj
        I should be able to cut down storage by >1000x. Would need to save <10GB

    """
    def __init__(self, state_size=(80,80,3), size=20e6, traj_len=100):
        self.size = int(size) #what, are you even going to fill this up?
        self.traj_len = traj_len

        self.obs_buffer = np.zeros((self.size, traj_len, state_size[0], 
            state_size[1], state_size[2]))
        self.pos_buffer = np.zeros((self.size, traj_len, 3))
        self.dir_buffer = np.zeros((self.size, traj_len))
        self.num_elements = 0

    def add(self, obs, pos, dirs):
        assert obs.shape[0] == pos.shape[0]
        assert obs.shape[0] == dirs.shape[0]
        batch_size = obs.shape[0]
        self.obs_buffer[self.num_elements:self.num_elements+batch_size] = obs
        self.pos_buffer[self.num_elements:self.num_elements+batch_size] = pos
        self.dir_buffer[self.num_elements:self.num_elements+batch_size] = dirs
        self.num_elements += batch_size

        # TODO: Check if buffer is full

    def sample(self, sample_size):
        if sample_size > self.num_elements:
            print("Ah! Should be more elements in buffer first!")
        print("shape of samples:")
        indecies = np.random.choice(self.num_elements, size=sample_size, 
            replace=True)

        obs = np.array([self.obs_buffer[i] for i in indecies])
        pos = np.array([self.pos_buffer[i] for i in indecies])
        dirs = np.array([self.dir_buffer[i] for i in indecies])

        return obs, pos, dirs


class BigReplayBuffer(object):
    """
    Replay Buffer that saves storage space by only saving the actions taken 
    in an environment and the seed for that round, such that it re-generate
    the same observations from the environment.

    The tradeoff is there is about 1 second of latency per sample, due to the 
    env.reset() bottleneck.

    TODO: Make work with ParallelEnv
    """
    def __init__(self, env, action_size=7, size=20e6, traj_len=100):
        self.size = int(size) #what, are you even going to fill this up?
        self.action_size = action_size
        self.traj_len = traj_len

        self.act_buffer = np.zeros((self.size, self.traj_len, self.action_size),
            dtype=np.intc)
        self.seed_buffer = np.zeros(self.size, dtype=int)
        self.num_elements = 0

        self.env = env
        self.env.reset()
        
        init_obs = self.env.observations()
        self.obs_types = init_obs.keys()
        self.obs_shapes = {}
        for obs_type in self.obs_types:
            self.obs_shapes[obs_type] = init_obs[obs_type].shape



    def add(self, acts, seeds):
        assert acts.shape[0] == seeds.shape[0]
        batch_size = acts.shape[0]

        print("seed buff: ", self.seed_buffer)

        if self.num_elements + batch_size >= self.size:
            # TODO: Fail better later hehe
            # assert(False, "Replay Buffer Full!")
            print("uh...")

        self.act_buffer[self.num_elements:self.num_elements+batch_size] = acts
        self.seed_buffer[self.num_elements:self.num_elements+batch_size] = seeds
        self.num_elements += batch_size
        
        print("seed buff: ", self.seed_buffer)



    def sample(self, sample_size):
        """ 
        Uniformly sample sample_size trajectories from the buffer. 
        Returns dict mapping
            obs_types -> np.array([sample_size, traj_length, *obs_shape])
        """
        if sample_size > self.num_elements:
            print("Ah! Should be more elements in buffer first!")
        print("shape of samples:")
        indecies = np.random.choice(self.num_elements, size=sample_size, 
            replace=False)

        obs_data = self._make_obs_structure(sample_size)

        for i, sample_index in enumerate(indecies):
            print("self.seed_buffer[sample_index]", self.seed_buffer[sample_index])
            self._regen_traj(self.act_buffer[sample_index], 
                self.seed_buffer[sample_index], i, obs_data) 

        return obs_data

    def _regen_traj(self, acts, seed, data_index, obs_data):
        self.env.reset(seed=seed)

        # Actually regen trajactories
        for t in range(self.traj_len):
            obs = self.env.observations()
            for obs_type in self.obs_types:
                obs_data[obs_type][data_index,t] = obs[obs_type]
            print(acts[t])
            print(obs['POS'])
            self.env.step(acts[t])
        return obs_data

    def _make_obs_structure(self, sample_size):
        """ 
        Make map from obs_type to np.zeros of shape 
        (sample_size, traj_len, *obs_shape)
        """
        all_obs = {}
        for obs_type in self.obs_types :
            new_shape = [sample_size, self.traj_len]
            new_shape.extend(list(self.obs_shapes[obs_type]))
            all_obs[obs_type] = np.zeros(new_shape)
        return all_obs


if __name__ == '__main__':
    pass