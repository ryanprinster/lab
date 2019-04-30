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
import random
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

  def step(self, scale):
    """Choose a random amount of a randomly selected action."""
    action_choice = random.randint(0, self.action_count - 1)
    action_amount = random.randint(self.action_spec[action_choice]['min'],
                                   self.action_spec[action_choice]['max'])
    action = np.zeros([self.action_count], dtype=np.intc)
    action[3] = 1*scale
    print("action:", action)
    return action

def randomTurn(mu, sigma, samples=1):
  return np.random.normal(mu, sigma, samples)

def randomVelocity(b, samples=1):
  return np.random.rayleigh(b, samples)

# def velAngToAction(vel, desired_velocity, aWall, desired_angle):
#   forward_action = 1
#   if desired_velocity < vel - 100 or desired_velocity <= 0:
#     forward_action = 0

#   deg_per_pixel = .1043701171875 # degrees per pixel rotated
#   turn_angle = desired_angle - aWall 
#   pixels_to_turn = int(turn_angle / deg_per_pixel)
#   print("unclipped pixels_to_turn:", pixels_to_turn)
#   if pixels_to_turn > 512 or pixels_to_turn < -512:
#     print(pixels_to_turn)
#   # pixels_to_turn = np.clip(pixels_to_turn, -512, 512)
#   print("aWall: ", aWall)
#   print("desired_angle: ", desired_angle)

#   action = np.array([pixels_to_turn, 0, 0, forward_action, 0, 0, 0], 
#     dtype=np.intc)
#   return action

def run(width, height, level_script, frame_count):
  """Spins up an environment and runs the random agent."""
  threshold_distance = 16 #currently abitrary number
  threshold_angle = 90
  mu = 0 #currently abitrary number
  sigma = 12 #currently abitrary number
  b = 1 #currently abitrary number


  config = {'width': str(width), 'height': str(height)}
  observations = ['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
                'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES']
  env = deepmind_lab.Lab(level_script, observations, config=config)
  env.reset()

  observation_data = []
  position_data = []
  direction_data = []


  for i in range(frame_count):
    dWall = env.observations()['DISTANCE_TO_WALL']
    aWall = env.observations()['ANGLE_TO_WALL']
    vel = abs(env.observations()['VEL.TRANS'][1])
    pos = env.observations()['POS']
    yaw = env.observations()['ANGLES'][1]
    obs = env.observations()['RGB_INTERLEAVED']

    print("angles: ", env.observations()['ANGLES'])
    print("pos: ", pos)

    # Update
    if dWall < threshold_distance and abs(aWall) < threshold_angle and aWall <= 360:
      # If heading towards a wall, slow down and turn away from it
      desired_angle = np.sign(aWall)*(threshold_angle-abs(aWall)) \
                      + randomTurn(mu, sigma)      
      deg_per_pixel = .1043701171875 # degrees per pixel rotated
      turn_angle = desired_angle - aWall 
      pixels_to_turn = int(turn_angle / deg_per_pixel)

      forward_action = 0
      prob_speed = dWall / threshold_distance
      if random.uniform(0, 1) < prob_speed:
        forward_action = 1

      action = np.array([pixels_to_turn, 0, 0, forward_action, 0, 0, 0], 
          dtype=np.intc)
    else:
      # Otherwise, act somewhat randomly
      desired_turn = randomTurn(mu, sigma)
      desired_velocity = randomVelocity(b)
      pixels_to_turn = int(desired_turn / .1043701171875)
      action = np.array([pixels_to_turn, 0, 0, 1, 0, 0, 0], 
        dtype=np.intc)

    env.step(action)

    observation_data.append(obs)
    position_data.append(pos)
    direction_data.append(yaw)

    print("action: ", action)
  # print("position_data: ", position_data)
  np.save('/mnt/hgfs/ryanprinster/test/obs_data.npy', np.array(observation_data))
  np.save('/mnt/hgfs/ryanprinster/test/pos_data.npy', np.array(position_data))
  np.save('/mnt/hgfs/ryanprinster/test/dir_data.npy', np.array(direction_data))


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
