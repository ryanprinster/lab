from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import itertools
import sys

from multiprocessing import Process, Queue, Pipe

import deepmind_lab

class ParallelEnv(object):
    def __init__(self, level_script, obs_types, config, num_envs):
        # TODO: Create ENUMS?
        self.level_script = level_script
        self.obs_types = obs_types
        self.config = config
        self.num_envs = num_envs

        self.pipes = [Pipe() for i in range(self.num_envs)]
        self.parent_conns, self.child_conns = zip(*self.pipes)
        self.processes = \
            [Process(target=self._env_worker, 
                args=(self.child_conns[i],self.level_script, self.obs_types, \
                    self.config)) 
            for i in range(self.num_envs)]

        for process in self.processes:
            process.start()

    def _env_worker(self, child_conn, level_script, obs_types, config):
        print("ParallelEnv._env_worker")
        env = deepmind_lab.Lab(level_script, obs_types, config=config)
        env.reset()

        while True:
            # data is a dict mapping inputs to values.
            flag, data = child_conn.recv()
            if flag == 'RESET':
                env.reset()
                package = True
            elif flag == 'OBSERVATIONS':
                package = env.observations()
            elif flag == 'STEP':
                package = env.step(data['action'])
            else:
                # PANIC!
                package = False
            child_conn.send(package)

    def _send_then_recv(self, packages):
        for i, conn in enumerate(self.parent_conns):
            conn.send(packages[i])
        return [conn.recv() for conn in self.parent_conns]

    def reset(self):
        # reset each env.
        packages = [('RESET', {}) for i in range(self.num_envs)]
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

if __name__ == '__main__':
    pass