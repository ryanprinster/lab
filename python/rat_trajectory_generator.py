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
import sys

class RatTrajectoryGenerator(object):
    def __init__(self, env, trajectory_length=100):
        self.env = env
        self.frame_count = trajectory_length
        self.threshold_distance = 16 #currently abitrary number
        self.threshold_angle = 90
        self.mu = 0 #currently abitrary number
        self.sigma = 12 #currently abitrary number
        self.b = 1 #currently abitrary number
        # self.reset()

        # Could use attributes to generate more data on each run
        # self.observation_data = []
        # self.position_data = []
        # self.direction_data = []
        # self.trans_velocity_data = []
        # self.ang_velocity_data = []
        # self.env.reset()
        print('Finishing RatTrajectoryGenerator __init__')
        sys.stdout.flush()

    def randomTurn(self, samples=1):
        return np.random.normal(self.mu, self.sigma, samples)

    def randomVelocity(self, samples=1):
        return np.random.rayleigh(self.b, samples)

    def generateTrajectories(self):
        """ Generates a batch of trajectories, one for each parallel env """
        print("Generating Trajectories")
        sys.stdout.flush()

        seeds = [random.randint(-sys.maxint - 1, sys.maxint) for i in range(self.env.num_envs)]
        self.env.reset(seed=seeds)

        observations = []
        positions = []
        directions = []
        trans_velocitys = []
        strafe_trans_velocitys = []
        ang_velocitys = []
        actions = []

        prev_yaw = 0
        for i in range(self.frame_count):
            dWall = self.env.observations()['DISTANCE_TO_WALL']
            aWall = self.env.observations()['ANGLE_TO_WALL']
            vel = abs(self.env.observations()['VEL.TRANS'][:,1])
            s_vel = self.env.observations()['VEL.TRANS'][:,0]
            pos = self.env.observations()['POS']
            yaw = self.env.observations()['ANGLES'][:,1]
            obs = self.env.observations()['RGB_INTERLEAVED']
            # Note: On the lua side, game:playerInfo().anglesVel only works
            # during :game human playing for some reason
            ang_vel = yaw - prev_yaw # in px/frame
            prev_yaw = yaw

            def update_traj(dWall, aWall, vel, pos, yaw, obs):
                # Update
                if dWall < self.threshold_distance and abs(aWall) < self.threshold_angle and aWall <= 360:
                    # If heading towards a wall, slow down and turn away from it
                    desired_angle = np.sign(aWall)*(self.threshold_angle-abs(aWall)) \
                                  + self.randomTurn()      
                    deg_per_pixel = .1043701171875 # degrees per pixel rotated
                    turn_angle = desired_angle - aWall 
                    pixels_to_turn = int(turn_angle / deg_per_pixel)

                    forward_action = 0
                    prob_speed = dWall / self.threshold_distance
                    if random.uniform(0, 1) < prob_speed:
                        forward_action = 1

                    action = np.array([pixels_to_turn, 0, 0, forward_action, 0, 0, 0], 
                        dtype=np.intc)
                else:
                    # Otherwise, act somewhat randomly
                    desired_turn = self.randomTurn()
                    desired_velocity = self.randomVelocity()
                    pixels_to_turn = int(desired_turn / .1043701171875)
                    action = np.array([pixels_to_turn, 0, 0, 1, 0, 0, 0], 
                        dtype=np.intc)

                return action, obs, pos, yaw, vel
            
            data = [update_traj(dWall[j], aWall[j], vel[j], pos[j], yaw[j], obs[j]) \
                for j in range(self.env.num_envs)]
            action, obs, pos, yaw, vel = zip(*data)

            self.env.step(action)

            observations.append(obs)
            positions.append(pos)
            directions.append(yaw)
            trans_velocitys.append(vel)
            strafe_trans_velocitys.append(s_vel)
            ang_velocitys.append(ang_vel)
            actions.append(action)

        observations = np.swapaxes(observations, 0, 1)
        positions = np.swapaxes(positions, 0, 1)
        directions = np.swapaxes(directions, 0, 1)
        trans_velocitys = np.swapaxes(trans_velocitys, 0, 1)
        strafe_trans_velocitys = np.swapaxes(strafe_trans_velocitys, 0, 1)
        ang_velocitys = np.swapaxes(ang_velocitys, 0, 1)
        actions = np.swapaxes(actions, 0, 1)
        seeds = np.array(seeds)

        return (observations, positions, directions, trans_velocitys, \
            strafe_trans_velocitys, ang_velocitys, actions, seeds)# hackyyy

    def generateAboutNTrajectories(self, N):
        # TODO: Change to exactly N trajectories
        observation_data = []
        position_data = []
        direction_data = []
        trans_velocity_data = []
        strafe_trans_velocity_data = []
        ang_velocity_data = []
        actions_data = []
        seeds_data = []

        for i in range(int(N/self.env.num_envs)):
            obs, pos, dire, trans_vel, s_trans_vel, ang_vel, actions, seeds = \
            self.generateTrajectories()

            observation_data.append(obs)
            position_data.append(pos)
            direction_data.append(dire)
            trans_velocity_data.append(trans_vel)
            strafe_trans_velocity_data.append(s_trans_vel)
            ang_velocity_data.append(ang_vel)
            actions_data.append(actions)
            seeds_data.append(seeds)

        observation_data = np.concatenate(np.array(observation_data), axis=0)
        position_data = np.concatenate(np.array(position_data), axis=0)
        direction_data = np.concatenate(np.array(direction_data), axis=0)
        trans_velocity_data = np.concatenate(np.array(trans_velocity_data), \
            axis=0)
        strafe_trans_velocity_data = np.concatenate(np.array(strafe_trans_velocity_data), \
            axis=0)
        ang_velocity_data = np.concatenate(np.array(ang_velocity_data), axis=0)
        actions_data = np.concatenate(np.array(actions_data), axis=0)
        seeds_data = np.concatenate(np.array(seeds_data), axis=0)

        return (observation_data, position_data, direction_data,
            trans_velocity_data, strafe_trans_velocity_data, ang_velocity_data), actions_data, seeds_data


if __name__ == '__main__':
    pass