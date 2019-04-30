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

from multiprocessing import Process, Queue, Pipe

import deepmind_lab

import tensorflow as tf



class GridNetwork(object):
    """Grid Network"""
    def __init__(self, name, lstm_size=128, grid_layer_size=512, 
        N=256, M=12, learning_rate = 1e-5, grad_clip_thresh=1e-5, max_time=100):
        with tf.variable_scope(name):
            
            # Constructing inputs to the lstm layer
            self.translational_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='trans_vel_input')
            self.sine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='sin_ang_vel_input')
            self.cosine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='cos_ang_vel_input')

            self.lstm_input = tf.concat([
                self.translational_velocity,
                self.sine_angular_velocity,
                self.cosine_angular_velocity,],
                axis=2, name='lstm_input')

            self.place_activity = tf.placeholder(tf.float32, [None, N], 
                name='place_activity_init')
            self.head_dir_activity = tf.placeholder(tf.float32, [None, M],
                name='head_activity_init')

            # Initial cell state and hidden state are linear transformations of 
            # place and head direction cells. Cell state and hidden state are 
            # concatenated for LSTMCell(state_is_tuple=False).

            # TODO: lstm_size/2 is arbitraty. should probably be proportion to N and M.
            # Actually should be same size as y and z?
            self.lstm_init_cell_hidden_state = tf.concat([
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                ], axis=1, name='lstm_init_cell_hidden_state')

            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)

            # TODO: unroll lstm for 100 time steps (paper says so)
            self.lstm_output, self.lstm_hidden_state_output = tf.nn.dynamic_rnn(
                self.lstm_cell,
                self.lstm_input,
                initial_state=self.lstm_init_cell_hidden_state,
                dtype=tf.float32)
            
            # Grid Cell layer
            self.grid_cell_layer = tf.contrib.layers.fully_connected(\
                self.lstm_output, grid_layer_size, activation_fn=None)

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            self.dropout = tf.contrib.layers.dropout(self.grid_cell_layer, 
                keep_prob=self.dropout_prob)

            # Place and head cells
            self.pred_place_cell_logits = tf.contrib.layers.fully_connected(\
                self.dropout, N)

            self.pred_head_dir_logits = tf.contrib.layers.fully_connected(\
                self.dropout, M)

            # self.pred_place_cell = tf.contrib.layers.softmax(self.pred_place_cell_logits)
            # self.pred_head_dir = tf.contrib.layers.softmax(self.pred_head_dir_logits)

            # Construct loss
            self.place_cell_labels = tf.placeholder(tf.float32, [None, max_time, N],
                name='place_cell_labels')
            self.head_dir_labels = tf.placeholder(tf.float32, [None, max_time, M],
                name='head_dir_labels')

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.math.add(self.place_cell_loss, self.head_dir_loss)


            # Optimizer
            # TODO: Add weight decay to decoder layers
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=.9)
            
            # Clip gradients
            gvs = self.optimizer.compute_gradients(self.loss)
            # Note: currently, last 6 ones here correspond to the grid cell layer
            # and the place/head cell layers. 
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip_thresh, grad_clip_thresh), var) for grad, var in gvs[-6:]]
            new_gvs = gvs[:-6] + capped_gvs

            self.train_op = self.optimizer.apply_gradients(new_gvs)

    # def train_step(self):


class PlaceCells(object):
    def __init__(self,N):
        self.N = N
        self.locations = self._generate_place_cells()
        self.sigma = 1 #NOTE: this can't be too small, or you'll get NaN errors
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
        activations = []
        for mu in self.locations:
            activation = self._gaussian(x, mu, self.sigma)
            activations.append(activation)
        normalized_activations = activations/np.sum(np.array(activations))
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


class RatTrajectoryGenerator():
    def __init__(self, env, trajectory_length=100):
        self.env = env
        self.frame_count = trajectory_length
        self.threshold_distance = 16 #currently abitrary number
        self.threshold_angle = 90
        self.mu = 0 #currently abitrary number
        self.sigma = 12 #currently abitrary number
        self.b = 1 #currently abitrary number
        self.reset()

    def reset(self):
        self.observation_data = []
        self.position_data = []
        self.direction_data = []
        self.trans_velocity_data = []
        self.ang_velocity_data = []
        self.env.reset()

    def randomTurn(self, samples=1):
        return np.random.normal(self.mu, self.sigma, samples)

    def randomVelocity(self, samples=1):
        return np.random.rayleigh(self.b, samples)

    def generateTrajectory(self):
        # TODO: Start at random location?
        self.env.reset()

        observations = []
        positions = []
        directions = []
        trans_velocitys = []
        ang_velocitys = []

        prev_yaw = 0
        for i in range(self.frame_count):
            dWall = self.env.observations()['DISTANCE_TO_WALL']
            aWall = self.env.observations()['ANGLE_TO_WALL']
            vel = abs(self.env.observations()['VEL.TRANS'][1])
            pos = self.env.observations()['POS']
            yaw = self.env.observations()['ANGLES'][1]
            obs = self.env.observations()['RGB_INTERLEAVED']
            # vel_rot = self.env.observations()['VEL.ROT'][1]
            # Note: On the lua side, game:playerInfo().anglesVel only works
            # during :game human playing for some reason
            ang_vel = yaw - prev_yaw # in px/frame
            prev_yaw = yaw

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

            self.env.step(action)

            observations.append(obs)
            positions.append(pos)
            directions.append(yaw)
            trans_velocitys.append(vel)
            ang_velocitys.append(ang_vel)

            # TODO: Conversions to all correct units?
        return observations, positions, directions, \
            trans_velocitys, ang_velocitys

    def generateBatchOfTrajectories(self, batch_size):
        self.reset()

        for i in range(batch_size):
            obs, pos, dire, trans_vel, ang_vel = self.generateTrajectory()
            self.observation_data.append(obs)
            self.position_data.append(pos)
            self.direction_data.append(dire)
            self.trans_velocity_data.append(trans_vel)
            self.ang_velocity_data.append(ang_vel)

        return self.observation_data, self.position_data, self.direction_data, \
            self.trans_velocity_data, self.ang_velocity_data


class Trainer(object):
    def __init__(self, level_script, env, N=256, M=12, trajectory_length=100,
        train_iterations=2):
        self.place_cells = PlaceCells(N)
        self.head_cells = HeadDirCells(M)
        self.grid_network = GridNetwork(name="grid_network") # add this later
        self.level_script = level_script
        self.env = env
        self.train_iterations = train_iterations
        self.rat = RatTrajectoryGenerator(env, trajectory_length)

    def train(self):

        observation_data, position_data, direction_data, trans_velocity_data, \
            ang_velocity_data = self.rat.generateBatchOfTrajectories(10)
        
        cos_vel_data = np.expand_dims(np.cos(ang_velocity_data), axis=2)
        sin_vel_data = np.expand_dims(np.sin(ang_velocity_data), axis=2)
        x_y_positions = np.delete(position_data,2,axis=2)
        trans_vel_data = np.expand_dims(trans_velocity_data, axis=2)
        direction_data = np.array(direction_data)

        place_cell_activity_labels = \
            self.place_cells.get_batched_ground_truth_activations(x_y_positions)
        head_cell_activity_labels = \
            self.head_cells.get_batched_ground_truth_activations(direction_data)

        initial_pos_activity = place_cell_activity_labels[:, 0, :]
        initial_dir_activity = head_cell_activity_labels[:, 0, :]

        # Input data
        print("cos_vel_data: ", cos_vel_data.shape)
        print("sin_vel_data: ", sin_vel_data.shape)
        print("trans_vel_data: ", trans_vel_data.shape)

        # Labels / activities
        print("x_y_positions: ", x_y_positions.shape)
        print("direction_data: ", direction_data.shape)        
        print("place_cell_activity_labels: ", place_cell_activity_labels.shape)
        print("head_cell_activity_labels: ", head_cell_activity_labels.shape)        

        # LSTM initiation
        print("initial_pos_activity: ", initial_pos_activity.shape)
        print("initial_dir_activity: ", initial_dir_activity.shape)

        # TODO: initial_pos_activity / place cells are showing NaNs again
        # TODO: move this stuff to grid cell

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            feed = {
                self.grid_network.translational_velocity: trans_velocity_data,
                self.grid_network.cosine_angular_velocity: cos_vel_data,
                self.grid_network.sine_angular_velocity: sin_vel_data,
                self.grid_network.place_activity: initial_pos_activity,
                self.grid_network.head_dir_activity: initial_dir_activity,
                self.grid_network.dropout_prob: .5,
                self.grid_network.place_cell_labels: place_cell_activity_labels,
                self.grid_network.head_dir_labels: head_cell_activity_labels,
            }

            _, loss = sess.run(
                [self.grid_network.train_op, self.grid_network.loss], 
                feed_dict=feed)
            print("loss: ", loss)






def run(width, height, level_script, frame_count):
    """Spins up an environment and runs the agent."""
    config = {'width': str(width), 'height': str(height)}
    obs_types = ['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
                'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES']
    env = deepmind_lab.Lab(level_script, obs_types, config=config)

    learning_rate = 1e-5
    train_iterations = 2
    N=256
    M=12
    trajectory_length=100

    # place_cells = PlaceCells(N)
    # head_dir_cells = HeadDirCells(M)

    # grid_network = GridNetwork(name="grid_network")
    # rat = RatTrajectoryGenerator(env)

    trainer = Trainer(level_script, env)
    trainer.train()




    # Testing stuff
    # print("place_cells: ", place_cells.locations)
    # print("head_dir_cells: ", head_dir_cells.directions)

    # loc = [450/32., 450/32.]
    # dir_ = 0
    # print("place cell activation wrt:", loc)
    # print("head dir cell activation wrt:", dir_)

    # print("place_cells_activations: ", 
    #     place_cells.get_ground_truth_activation(loc))
    # print("head_dir_cells_activations: ", 
    #     head_dir_cells.get_ground_truth_activation(dir_))

    # locs = [[450/32., 450/32.], [150/32., 150/32.]]
    # dirs = [0,1]

    # print("place_cells_activations: ", 
    #     place_cells.get_ground_truth_activations(locs))
    # print("head_dir_cells_activations: ", 
    #     head_dir_cells.get_ground_truth_activations(dirs))


    # print("obs_data shape:", obs_data.shape)
    # print("pos_data shape:", pos_data.shape)
    # print("dir_data shape:", dir_data.shape)




  


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
