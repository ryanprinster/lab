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
"""Grid Cell Agent"""
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
        N=256, M=12, learning_rate=1e-3, grad_clip_thresh=1e-5, max_time=100):

        # TODO: share these
        self.grid_layer_size = grid_layer_size


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

            self.optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=learning_rate, momentum=.9).minimize(self.loss)

            # # Optimizer
            # # TODO: Add weight decay to decoder layers
            # self.optimizer = tf.train.RMSPropOptimizer(\
            #     learning_rate=learning_rate, momentum=.9)
            
            # # Clip gradients
            # gvs = self.optimizer.compute_gradients(self.loss)
            # # Note: currently, last 6 ones here correspond to the grid cell layer
            # # and the place/head cell layers. 
            # capped_gvs = [(tf.clip_by_value(grad, -grad_clip_thresh, \
            #     grad_clip_thresh), var) \
            #     for grad, var in gvs[-6:]]

            # new_gvs = gvs[:-6] + capped_gvs

            # self.train_op = self.optimizer.apply_gradients(new_gvs)

            # Summary Stuff
            self.loss_summary = tf.summary.histogram('loss', self.loss)
            self.loss_summary = tf.summary.histogram('mean loss', tf.math.reduce_mean(self.loss))
            self.summary = tf.summary.merge_all()



    def _transform_input_data(self, data, place_cells, head_cells):
        observation_data, position_data, direction_data, trans_velocity_data, \
            ang_velocity_data = data 
        
        cos_vel_data = np.expand_dims(np.cos(ang_velocity_data), axis=2)
        sin_vel_data = np.expand_dims(np.sin(ang_velocity_data), axis=2)
        x_y_positions = np.delete(position_data,2,axis=2)
        trans_vel_data = np.expand_dims(trans_velocity_data, axis=2)
        direction_data = np.array(direction_data)

        place_cell_activity_labels = \
            place_cells.get_batched_ground_truth_activations(x_y_positions)
        head_cell_activity_labels = \
            head_cells.get_batched_ground_truth_activations(direction_data)

        initial_pos_activity = place_cell_activity_labels[:, 0, :]
        initial_dir_activity = head_cell_activity_labels[:, 0, :]

        # # Input data
        # print("cos_vel_data: ", cos_vel_data.shape)
        # print("sin_vel_data: ", sin_vel_data.shape)
        # print("trans_vel_data: ", trans_vel_data.shape)

        # # Labels / activities
        # print("x_y_positions: ", x_y_positions.shape)
        # print("direction_data: ", direction_data.shape)        
        # print("place_cell_activity_labels: ", place_cell_activity_labels.shape)
        # print("head_cell_activity_labels: ", head_cell_activity_labels.shape)        

        # # LSTM initiation
        # print("initial_pos_activity: ", initial_pos_activity.shape)
        # print("initial_dir_activity: ", initial_dir_activity.shape)

        return (trans_vel_data, cos_vel_data, sin_vel_data, initial_pos_activity, 
        initial_dir_activity, place_cell_activity_labels, 
        head_cell_activity_labels, x_y_positions)

    def train_step(self, sess, data, place_cells, head_cells):

        trans_vel_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            self._transform_input_data(data, place_cells, head_cells)

        feed = {
            self.translational_velocity: trans_vel_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.place_activity: initial_pos_activity,
            self.head_dir_activity: initial_dir_activity,
            self.dropout_prob: .5,
            self.place_cell_labels: place_cell_activity_labels,
            self.head_dir_labels: head_cell_activity_labels,
        }

        _, loss, summary = sess.run(
            [self.optimizer, self.loss, self.summary], 
            feed_dict=feed)

        return loss, summary

    def get_grid_layer_activations(self, sess, data, place_cells, head_cells):

        trans_vel_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            self._transform_input_data(data, place_cells, head_cells)

        feed = {
            self.translational_velocity: trans_vel_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.place_activity: initial_pos_activity,
            self.head_dir_activity: initial_dir_activity,
            self.dropout_prob: 0,
        }

        grid_layer_activations = sess.run(self.grid_cell_layer, feed_dict=feed)
        hist = self._grid_activations_to_histogram(grid_layer_activations, 
            x_y_positions)

        return hist

    def _grid_activations_to_histogram(self, grid_layer_activations, 
            x_y_positions, bins=32):

        grid_acts = np.reshape(grid_layer_activations, (-1, self.grid_layer_size))
        xy_pos = np.reshape(x_y_positions, (-1, 2))
        data = zip(xy_pos, grid_acts)        

        def xy_coord_to_ind(loc):
            # Assuming a valid location
            x_coord, y_coord = loc
            low, high = 100, 800
            x, y = x_coord-low, y_coord-low
            size_of_bin = (high-low)/bins
            x_ind, y_ind = int(x/(size_of_bin+1)), int(y/(size_of_bin+1)) #wont have anything in last bin really?
            return x_ind, y_ind

        def construct_histogram(data):
            #janky avoid /0
            counts = np.ones((self.grid_layer_size, bins, bins))*1e-10
            activations = np.zeros((self.grid_layer_size, bins, bins))
            for loc, acts in data:
                x_ind, y_ind = xy_coord_to_ind(loc)
                for i, act in enumerate(acts):
                    activations[i, x_ind, y_ind] += act
                    counts[i, x_ind, y_ind] += 1
            histograms = activations/counts
            return histograms

        return construct_histogram(data)

    # TODO: similar function for head direction cells 


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
        x = x/32. # convert grid world unnits
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

    def randomTurn(self, samples=1):
        return np.random.normal(self.mu, self.sigma, samples)

    def randomVelocity(self, samples=1):
        return np.random.rayleigh(self.b, samples)

    def generateTrajectories(self):
        """ Generates a batch of trajectories, one for each parallel env """
        print("Generating Trajectories")

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
            vel = abs(self.env.observations()['VEL.TRANS'][:,1])
            pos = self.env.observations()['POS']
            yaw = self.env.observations()['ANGLES'][:,1]
            obs = self.env.observations()['RGB_INTERLEAVED']
            # vel_rot = self.env.observations()['VEL.ROT'][1]
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
            ang_velocitys.append(ang_vel)

        observations = np.swapaxes(observations, 0, 1)
        positions = np.swapaxes(positions, 0, 1)
        directions = np.swapaxes(directions, 0, 1)
        trans_velocitys = np.swapaxes(trans_velocitys, 0, 1)
        ang_velocitys = np.swapaxes(ang_velocitys, 0, 1)

        return (observations, positions, directions, trans_velocitys, \
            ang_velocitys) # hackyyy

    def generateAboutNTrajectories(self, N):
        observation_data = []
        position_data = []
        direction_data = []
        trans_velocity_data = []
        ang_velocity_data = []

        for i in range(int(N/self.env.num_envs)):
            obs, pos, dire, trans_vel, ang_vel = self.generateTrajectories()
            observation_data.append(obs)
            position_data.append(pos)
            direction_data.append(dire)
            trans_velocity_data.append(trans_vel)
            ang_velocity_data.append(ang_vel)

        observation_data = np.concatenate(np.array(observation_data), axis=0)
        position_data = np.concatenate(np.array(position_data), axis=0)
        direction_data = np.concatenate(np.array(direction_data), axis=0)
        trans_velocity_data = np.concatenate(np.array(trans_velocity_data), \
            axis=0)
        ang_velocity_data = np.concatenate(np.array(ang_velocity_data), axis=0)

        return (observation_data, position_data, direction_data,
            trans_velocity_data, ang_velocity_data)


class Trainer(object):
    def __init__(self, 
        base_path,
        unique_exp_name, # string indicating experiment
        restore=False,
        N=256, 
        M=12, 
        trajectory_length=100, 
        train_iterations=5000, 
        learning_rate=1e-3, 
        num_envs=4,
        level_script='tests/empty_room_test', 
        obs_types=['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
                'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES'],
        config={'width': str(80), 'height': str(80)}):

        self.env = ParallelEnv(level_script, obs_types, config, num_envs)
        self.place_cells = PlaceCells(N)
        self.head_cells = HeadDirCells(M)
        self.grid_network = GridNetwork(name="grid_network", 
            learning_rate=learning_rate, max_time=trajectory_length) # add this later
        self.rat = RatTrajectoryGenerator(self.env, trajectory_length)

        self.train_iterations = train_iterations

        self.base_path = base_path
        self.experiments_save_path = self.base_path + 'experiments/'
        self.unique_exp_save_path = self.experiments_save_path \
            + 'exp_' + unique_exp_name + '/'

        self.restore = restore
        self.saver = tf.train.Saver()

    def train(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if self.restore:
                self.saver.restore(sess, \
                    tf.train.latest_checkpoint(self.unique_exp_save_path))

            train_writer = tf.summary.FileWriter(self.unique_exp_save_path)
            """

            say N envs, N*10 trajectories / batch = 10s
            Nenvs* 100steps/sec = 100*N steps/s
            10 minutes to reach 1mil
            estimated this will take 40+ hours on 16 cpus w/ min bottleneck of 
            1s/ reset

            to reach 100 mil (16+ hours) need 1mil = iterations*batchsize
            """

            for i in range(self.train_iterations):
                print("Train iter: ", i)
                data = self.rat.generateAboutNTrajectories(16)
                loss, summary = self.grid_network.train_step(sess, data, self.place_cells, 
                    self.head_cells)
                train_writer.add_summary(summary, i)
                print("loss: ", np.mean(loss))

                if i % 10 == 0:
                    self.saver.save(sess, self.unique_exp_save_path)

            print("Seeing if there are grid cells:")
            data = self.rat.generateAboutNTrajectories(160)
            histograms = self.grid_network.get_grid_layer_activations(sess, 
                data, self.place_cells, self.head_cells)

            print("And we have histograms:")
            print(histograms.shape)

            np.save(self.unique_exp_save_path + 'place_cell_histograms.npy', \
                histograms)


def run(level_script, base_path, num_envs, learning_rate, exp_name):
    """Spins up an environment and runs the agent."""

    # TESTING:
    # base_path = '/mnt/hgfs/ryanprinster/data/'
    # num_envs = 4
    learning_rate = 1e-3
    trainerParallel = Trainer(
        base_path=base_path,
        unique_exp_name=exp_name,
        num_envs=num_envs,
        learning_rate=learning_rate)
    trainerParallel.train()
    # TODO - tensorboard stuff


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

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    run(args.level_script, args.base_path, args.num_envs, args.learning_rate, \
        args.exp_name)
