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
    def __init__(self, size=20e6):
        self.size = size #what, are you even going to fill this up?
        self.buffer = np.array([])

    def add(self, elements):
        # elements is an np.array of elements to add
        self.buffer = np.concatenate((self.buffer, elements), axis=0)

    def sample(self, sample_size):
        if sample_size < len(self.buffer):
            print("Ah! Should be more elements in buffer first!")
        print("shape of samples:")
        # Wont work with >1D array
        return np.random.choice(self.buffer, size=sample_size, replace=True)


if __name__ == '__main__':
    pass