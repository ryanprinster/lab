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

  def step(self):
    """Choose a random amount of a randomly selected action."""
    action_choice = random.randint(0, self.action_count - 1)
    action_amount = random.randint(self.action_spec[action_choice]['min'],
                                   self.action_spec[action_choice]['max'])
    action = np.zeros([self.action_count], dtype=np.intc)
    action[action_choice] = action_amount
    return action


def other_one(x):
  return x

def worker(child_conn, level_script, config, frame_count):

  env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config=config)
  env.reset()
  print("HAHAH I MADE ANOTHER ONE")
  
  x = other_one(10)

  reward = 0
  agent = RandomAgent(env.action_spec())
  for _ in six.moves.range(frame_count):
    if not env.is_running():
      print('Environment stopped early')
      env.reset()
      agent.reset()
    action = agent.step()
    reward += env.step(action, num_steps=1)

  print('Finished after %i steps. Total reward received is %f'
        % (frame_count, reward))
  return reward


def test(x):
  print(x)

def run(width, height, level_script, frame_count):
  """Spins up an environment and runs the random agent."""
  config = {'width': str(width), 'height': str(height)}

  env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config=config)

  print("starting env loop thing")
  import time
  start = time.time()
  for i in range(10):
    env.reset()
  end = time.time()

  print("Time for 100 resets: ", end-start)



  # num_processes = 10

  # pipes = [Pipe() for i in range(num_processes)]
  # parent_conns, child_conns = zip(*pipes)
  
  # processes = [Process(target=worker, 
  #             args=(child_conns[i],level_script, config, frame_count)) 
  #             for i in range(num_processes)]

  # # Initialize
  # for i in range(len(processes)):
  #     processes[i].start()
  #     package = (1, i)
  #     parent_conns[i].send(package)

  # responses = [parent_conns[i].recv() for i in range(num_processes)]
  
  # print(responses)

  # for i in range(num_processes):
  #   package = responses[i]
  #   parent_conns[i].send(package)

  


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
