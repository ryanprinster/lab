from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import itertools
import sys

class PlaceCells(object):
    def __init__(self,N,sigma=1):
        self.N = N
        self.locations = self._generate_place_cells()
        self.sigma = sigma #NOTE: this can't be too small, or you'll get NaN errors
        # Note: In DeepMind Lab there are 32 units to the meter, 
        # so we multiply any plane or position by this number.

    def _generate_place_cells(self):
        """
        Generates ground truth locations for place cells.
        TODO: do this for a given maze.
        """
        low, high = 100/32., 800/32.
        place_cell_locations = np.random.uniform(low, high, (self.N, 2)) # place cell centers
        return place_cell_locations

    def _gaussian(self, x, mu, sig):
        return np.exp(-np.sum(np.power(x - mu, 2.)) / (2 * np.power(sig, 2.)))
            # problem might be in the np.sum

    def get_ground_truth_activation(self, x):
        """For a given location, get activations of all place cells"""
        x = x/32. # convert grid world unnits
        activations = []
        for mu in self.locations:
            activation = self._gaussian(x, mu, self.sigma)
            activations.append(activation)
        normalized_activations = activations/np.sum(np.array(activations))
        if np.isnan(np.sum(normalized_activations)):
            print("NAN PROBLEM!")
            return np.zeros(normalized_activations.shape)
        return normalized_activations

    def get_ground_truth_activations(self, X):
        """ For a list of locations, get activations of all place cells"""
        many_activations = []
        for loc in X:
            many_activations.append(self.get_ground_truth_activation(loc))
        return many_activations

    def get_batched_ground_truth_activations(self, batch_x):
        return np.array([self.get_ground_truth_activations(X) for X in batch_x])

class HeadDirCells(object):
    def __init__(self, M):
        self.M = M
        self.directions= self._generate_head_cells(M)
        self.k = 20

    def _generate_head_cells(self, M):
        # just doing this in degrees
        # actually need to do this in radians
        low, high = -np.pi, np.pi
        head_cell_directions = np.random.uniform(low, high, self.M)
        return head_cell_directions

    def _von_mises(self, phi, mu):
        return np.exp(self.k * np.cos(phi-mu))

    def get_ground_truth_activation(self, phi):
        """For a given head direction, get activations of all head dir cells"""
        activations = []
        for mu in self.directions:
            activations.append(self._von_mises(phi, mu))
        return activations/np.sum(np.array(activations))

    def get_ground_truth_activations(self, X):
        """For a list of locations, get activationns of all head dir cells"""
        return [self.get_ground_truth_activation(phi) for phi in X]

    def get_batched_ground_truth_activations(self, batch_x):
        return np.array([self.get_ground_truth_activations(X) for X in batch_x])

if __name__ == '__main__':
    pass
