# Copyright 2016-17 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
################################################################################
"""Attempt at a3c"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six

from multiprocessing import Process, Queue, Pipe

import deepmind_lab

gamma = .99

def is_terminal(state):
    # Currently, is terminal if and only if you recieve a reward


def actor_learner_thread(child_conn, level_script, config):
    env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config=config)
    t = 1
    while T < T_max:
        d_theta, d_theta_v  = 0, 0
        theta, theta_v = child_conn.recv() # Get most current version
        t_start = t
        state = env.observations()['RGB_INTERLEAVED']
        while not is_terminal(state) or t-t_start == t_max:
            reward, next_state = env_step()
            t += 1
            T += 1
        R = 0 if is_terminal(state) else valNetwork(state) #Bootstrap?
        for i in range(t-1, t_start, -1):
            R = reward[i] + gamma*R #accumulate these
            # Accumulate gradients
            d_theta += 
            d_theta_v += 
        child_conn.send(d_theta, d_theta_v)



def run(width, height, level_script, frame_count):
  """Spins up an environment and runs the random agent."""
  config = {'width': str(width), 'height': str(height)}




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

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.width, args.height, args.level_script, args.frame_count)
