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

  def step(self, scale=1):
    """Choose a random amount of a randomly selected action."""
    action_choice = random.randint(0, self.action_count - 1)
    action_amount = random.randint(self.action_spec[action_choice]['min'],
                                   self.action_spec[action_choice]['max'])
    action = np.zeros([self.action_count], dtype=np.intc)
    action[3] = 1*scale
    print("action:", action)
    return action

def run(width, height, level_script, frame_count):
  """Spins up an environment and runs the random agent."""

  # TESTING LOCALLY:

  print(np.empty((1,2,3)))

  # config = {'width': str(width), 'height': str(height)}
  # level_script = 'tests/trivial_maze'
  # observations = ['RGB_INTERLEAVED']
  # env = deepmind_lab.Lab(level_script, observations, config=config)
  # env.reset()

  # agent = RandomAgent(env.action_spec())

  # while True:
  #   action = agent.step()
  #   print(env.step(action))
  #   print(env.num_steps())

  # print("DISTANCE_TO_WALL")
  # print(env.observations()['DISTANCE_TO_WALL'])
  # print("ANGLE_TO_WALL")
  # print(env.observations()['ANGLE_TO_WALL'])


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
