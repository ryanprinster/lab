from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import itertools
import sys
import tensorflow as tf

from multiprocessing import Process, Queue, Pipe
from trainers import Trainer, TrajectoryGenerator

"""
Manager for slurm jobs
"""
class SlurmManager(object):
    """
    Class meant to deal with running multiple jobs with different hyperparams 
    using Slurm.
    """

    def __init__(self, slurm_array_index, base_path):
        """
        For now, let us assume slurm_task_id in {0, num_hyperparam_combos-1}.
        """
        self.slurm_array_index = slurm_array_index
        self.base_path = base_path

    def run(self):
        # TODO: make a more elegant, scalable way to take in different 
        # sets of hyperparams for different tasks. But for now:

        """
        hyperparams = [
            learning_rate, 
            batch_size, 
            train_iterations, 
            num_envs, 
            sigma, 
        ]
        """

        # For GridCellAgentTrainer and Trainer
        hyperparams = [
            [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            [32],
            [100000],
            [16],
            [1],
        ]

        combinations = list(itertools.product(*hyperparams))
        print("Hyperparam combos: ", combinations)
        hyperparam_selection = combinations[self.slurm_array_index]
        print("Hyperparam selection: ", hyperparam_selection)
        trainerParallel = Trainer(
            base_path=self.base_path,
            unique_exp_name='job_' + str(self.slurm_array_index),
            learning_rate=hyperparam_selection[0],
            batch_size=hyperparam_selection[1],
            train_iterations=hyperparam_selection[2],
            num_envs=hyperparam_selection[3],
            sigma=hyperparam_selection[4])
        trainerParallel.train()

        # For TrajectoryGenerator
        # hyperparams = [
        #     [100],
        #     [100000],
        # ]

        # combinations = list(itertools.product(*hyperparams))
        # print("Hyperparam combos: ", combinations)
        # hyperparam_selection = combinations[self.slurm_array_index]
        # print("Hyperparam selection: ", hyperparam_selection)
        # trainer = TrajectoryGenerator(
        #     base_path=self.base_path,
        #     unique_exp_name='job_' + str(self.slurm_array_index),
        #     num_envs=hyperparam_selection[0],
        #     num_traj_to_generate=hyperparam_selection[0])
        # trainer.train()

def hyperparam_selector():
    list(itertools.product(*hyperparams))

class A2CSlurmManager(object):
    """
    Class meant to deal with running multiple jobs with different hyperparams 
    using Slurm.
    """

    def __init__(self, slurm_array_index, base_path):
        """
        For now, let us assume slurm_task_id in {0, num_hyperparam_combos-1}.
        """
        self.slurm_array_index = slurm_array_index
        self.base_path = base_path

    def run(self):
        # TODO: make a more elegant, scalable way to take in different 
        # sets of hyperparams for different tasks. But for now:

        """
        hyperparams = [
            learning_rate, 
            batch_size, 
            train_iterations, 
            num_envs, 
            sigma, 
        ]
        """

        # For GridCellAgentTrainer and Trainer
        hyperparams = [
            [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            [32],
            [100000],
            [16],
            [1],
        ]

        combinations = list(itertools.product(*hyperparams))
        print("Hyperparam combos: ", combinations)
        hyperparam_selection = combinations[self.slurm_array_index]
        print("Hyperparam selection: ", hyperparam_selection)
        trainerParallel = Trainer(
            base_path=self.base_path,
            unique_exp_name='job_' + str(self.slurm_array_index),
            learning_rate=hyperparam_selection[0],
            batch_size=hyperparam_selection[1],
            train_iterations=hyperparam_selection[2],
            num_envs=hyperparam_selection[3],
            sigma=hyperparam_selection[4])
        trainerParallel.train()

def run(slurm_array_index, base_path):

    # TESTING LOCALLY:
    # base_path = '/mnt/hgfs/ryanprinster/data/'
    # slurm_array_index = 0

    SlurmManager(slurm_array_index, base_path).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame_count', type=int, default=10000,
                      help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
    parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str, default='tests/empty_room_test',
                      help='The environment level script to load')
    parser.add_argument('--base_path', type=str, default='/om/user/prinster/lab/my_data/',
                      help='base_path')
    parser.add_argument('--num_envs', type=str, default=16,
                      help='num environments to run in parallel')
    parser.add_argument('--learning_rate', type=str, default=1e-3,
                      help='learning_rate')
    parser.add_argument('--exp_name', type=str, default='test',
                      help='exp_name')
    parser.add_argument('--slurm_array_index', type=int, default=0,
                      help='id provided by slurm for which experiment to run')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    run(args.slurm_array_index, args.base_path)