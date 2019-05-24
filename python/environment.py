from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import itertools
import sys

from multiprocessing import Process, Queue, Pipe

import deepmind_lab

# class Util(object):
#     """
#     Contains functions related to the dmlab environment
#     """
#     def __init__(self):
#         pass

def _action(*entries):
    return np.array(entries, dtype=np.intc)

def map_to_dmlab(action_index):
    DMLAB_ACTIONS = [_action(-20, 0, 0, 0, 0, 0, 0),
    _action(20, 0, 0, 0, 0, 0, 0),
    _action(0, 0, -1, 0, 0, 0, 0),
    _action(0, 0, 1, 0, 0, 0, 0),
    _action(0, 0, 0, 1, 0, 0, 0),
    _action(0, 0, 0, -1, 0, 0, 0)]

    if isinstance(action_index, int):
        return DMLAB_ACTIONS[action_index]
    elif isinstance(action_index, list) or isinstance(action_index, np.ndarray):
        return np.array([DMLAB_ACTIONS[i] for i in action_index])
    else:
        print("Panic! Type is actually: ", type(action_index))



class ParallelEnv(object):
    """
    Mimics the Deepmind Lab Env API, for multiple environments in parallel.
    """
    # TODO: Add num_steps functionality
    def __init__(self, level_script, obs_types, config, num_envs, num_steps=1):
        # TODO: Create ENUMS?
        self.level_script = level_script
        self.obs_types = obs_types
        self.config = config
        self.num_envs = num_envs
        self.num_steps = num_steps

        self.pipes = [Pipe() for i in range(self.num_envs)]
        self.parent_conns, self.child_conns = zip(*self.pipes)
        self.processes = \
            [Process(target=self._env_worker, 
                args=(self.child_conns[i],self.level_script, self.obs_types, \
                    self.config)) 
            for i in range(self.num_envs)]

        for process in self.processes:
            process.start()
        print("Finished ParallelEnv __init__")
        sys.stdout.flush()

    def _env_worker(self, child_conn, level_script, obs_types, config):
        print("ParallelEnv._env_worker")
        sys.stdout.flush()
        
        env = deepmind_lab.Lab(level_script, obs_types, config=config)
        env.reset()

        while True:
            # data is a dict mapping inputs to values.
            flag, data = child_conn.recv()
            if flag == 'RESET':
                env.reset(seed=data['seed'])
                package = True
            elif flag == 'OBSERVATIONS':
                package = env.observations()
            elif flag == 'STEP':
                # requires dtype=np.intc
                package = env.step(data['action'], self.num_steps)
            elif flag == 'NUM_STEPS':
                package = env.num_steps()
            else:
                # PANIC!
                package = False
            child_conn.send(package)

    def _send_then_recv(self, packages):
        for i, conn in enumerate(self.parent_conns):
            conn.send(packages[i])
        return [conn.recv() for conn in self.parent_conns]

    def reset(self, seed=None):
        # reset each env.
        if seed is None:
            seed = [None for i in range(self.num_envs)]
        packages = [('RESET', {'seed': seed[i]}) for i in range(self.num_envs)]
        return self._send_then_recv(packages)

    def observations(self):
        # return a dict, mapping observation types to arrays of observations
        packages = [('OBSERVATIONS', {}) for i in range(self.num_envs)]
        data = self._send_then_recv(packages)
        result = {}
        for obs_type in self.obs_types:
            result[obs_type] = np.array(
                [data[i][obs_type] for i in range(self.num_envs)])
        return result

    def step(self, action):
        # takes an array of actions, returns an array of rewards
        packages = [('STEP', {'action': action[i]}) for i in \
            range(self.num_envs)]
        return np.array(self._send_then_recv(packages))

    def num_steps(self):
        # returns number of steps since last reset call
        packages = [('NUM_STEPS', {}) for i in range(self.num_envs)]
        return np.array(self._send_then_recv(packages))

if __name__ == '__main__':
    pass