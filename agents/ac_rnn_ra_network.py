# -*- coding: utf-8 -*-

"""Advantage actor-critic networks

This module contains computation graph ops to compose advantage actor-
critic networks.  Typically a global network will be constructed as well
as several worker networks, which sync with the global network and
propose updates asynchronously.

Todo:
    * Derive this class from AC Network base class.
    * Avoid using hard-coded name 'global' for global AC network.

"""

from re import match
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from util import normalized_columns_initializer


class AC_rnn_ra_Network():
    """Advantage actor-critic network with continuous policy an rnn.
    
    The rnn is implemented using LSTM cells with layer normalization,
    and accepts addition features, e.g., rewards and actions, for
    meta-learning.
    
    """
    
    def __init__(self,s_shape,a_size,scope,trainer,hlvl=0):
        """Initializes network graph ops with the desired scope
        
        Args:
            s_shape (list): the shape of the state tensor.
            a_size (int): the dimension of the continuous action vector.
            scope (str): scope for the tensorflow graph. Currently, the master
                A3C graph nodes must use 'global' as the scope, but worker A3C
                agents may be instantiated with any scope name.
            trainer: a tensorflow optimizer from the tf.train module.
            hlvl (int): the hierarchy level (starting at 0) of the agent
        
        """
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None]+list(s_shape),
                                         dtype=tf.float32)
            self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None,a_size],
                                               dtype=tf.float32)
            self.is_training_ph = tf.placeholder(tf.bool)
            with slim.arg_scope([slim.conv2d,slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.0), 
                                activation_fn=tf.nn.elu): # ARA - tf.nn.relu
                self.conv1 = slim.conv2d(inputs=self.inputs,num_outputs=16,
                                         kernel_size=[8,8],stride=[4,4],
                                         padding='VALID',scope='c1')
                self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,
                                         kernel_size=[4,4],stride=[2,2],
                                         padding='VALID',scope='c2')
                hidden = slim.fully_connected(slim.flatten(self.conv2),256,
                                              scope='fc1')
            #    
            hidden = tf.concat(axis=1,values=[hidden,self.prev_rewards,
                                              self.prev_actions])
            
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(256) # ARA - v1.0
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in) # ARA - v1.0
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,
                sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            # Output layer for value estimation
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            # Output layers for continuous policy estimation
            self.mu = slim.fully_connected(rnn_out,a_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None) # shape [batch, a_size]
            self.sig2 = slim.fully_connected(rnn_out,1,
                activation_fn=tf.nn.softplus,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None) # shape [batch, 1]
            # copy std's across columns [batch, a_size]
            self.stdev = tf.tile(tf.sqrt(self.sig2),[1,a_size])
            self.policy_dist = tf.contrib.distributions.MultivariateNormalDiag(
                self.mu,self.stdev)
            self.sample_a = tf.reshape(self.policy_dist.sample(1),[-1])
            
            # Only the worker network need ops for loss functions and gradient updating.
            # if scope != 'global':            
            if not match(r'global',scope):
                self.actions = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                # Loss functions
                self.value_loss = tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = tf.reduce_sum(self.policy_dist.entropy())
                self.policy_loss = -tf.reduce_sum(
                    self.policy_dist.log_pdf(self.actions)*self.advantages)
                self.loss = self.value_loss+self.policy_loss-self.entropy*5e-5

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,
                                                               40)
                
                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global_'+str(hlvl))
                self.apply_grads = trainer.apply_gradients(zip(grads,
                                                               global_vars))

