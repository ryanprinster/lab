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
"""A simple example of a random agent in deepmind_lab."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import sys
# from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.multiprocessing import ProcessingPool as Pool
# # pip install pathos
# import pathos

from multiprocessing import Process, Queue, Pipe

import deepmind_lab


class RandomAgent(object):
  """Basic random agent for DeepMind Lab."""
  def __init__(self, action_spec):
    self.action_spec = action_spec
    self.action_count = len(action_spec)

  def step(self):
    """Choose a random amount of a randomly selected action."""
    action_choice = random.randint(0, self.action_count - 1)
    action_amount = random.randint(self.action_spec[action_choice]['min'],
                                   self.action_spec[action_choice]['max'])
    action = np.zeros([self.action_count], dtype=np.intc)
    action[action_choice] = action_amount
    return action

def run(width, height, level_script, frame_count):
  """Spins up an environment and runs the random agent."""

  # TESTING LOCALLY:

  from replay_buffer import BigReplayBuffer
  from rat_trajectory_generator import RatTrajectoryGenerator as Rat
  from environment import ParallelEnv
  config={'width': str(80), 'height': str(80)}
  obs_types = ['RGB_INTERLEAVED', 'ANGLES', 'POS', 'VEL.TRANS', 'ANGLE_TO_WALL', 'DISTANCE_TO_WALL']

  seed=[-1,0,1,sys.maxint]
  env = ParallelEnv('tests/empty_room_test', obs_types, config, 4)
  env.reset(seed=seed)
  print("I think it worked?")
  print(env.observations()['POS'])

  rat = Rat(env, 100)
  data, actions, seeds = rat.generateAboutNTrajectories(16)
  print("actions", actions.shape)
  print("seeds", seeds.shape)
  print(seeds)
  env.reset(seed=seeds)
  
  
  # env = deepmind_lab.Lab('tests/empty_room_test', obs_types, config=config)

  # agent = RandomAgent(env.action_spec())
  # actions = []
  # for i in range(100):
  #   print(env.observations()['POS'])
  #   action = agent.step()
  #   print(action)
  #   env.step(action)
  #   actions.append(action)
  #   print


  # replay_buffer = BigReplayBuffer(env, size=100)
  # replay_buffer.add(np.array([actions]), np.array([seed]))
  # print(replay_buffer.sample(1)['POS'])


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
