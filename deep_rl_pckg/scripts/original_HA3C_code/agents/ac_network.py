# -*- coding: utf-8 -*-

"""Advantage actor-critic networks

This module contains computation graph ops to compose advantage actor-
critic networks.  Typically a global network will be constructed as well
as several worker networks, which sync with the global network and
propose updates asynchronously.

Todo:
    * Develop a common AC network interface. Consider an abstract base
      class with multiple derived classes to add specialized features.
    * Separately encapsulate major layers, e.g., input, rnn, output.
    * Avoid using hard-coded name 'global' for global AC network.

"""

from re import match
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from util import normalized_columns_initializer


class AC_Network():
    """Simple Advantage actor-critic network with discrete policy.
    
    This implementation includes only convolutional input layers and a
    fully-connected layer followed by linear output layers.
    
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
        
        """
        global_name = 'global_'+str(hlvl)

        #Create the master network, if it does not already exist
        if scope != global_name and \
           not tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_name):
            AC_Network(s_shape,a_size, global_name, None, hlvl)
        
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None]+list(s_shape),
                                         dtype=tf.float32)
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
            
            # Output layer for value estimation
            self.value = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            # Output layer for discrete policy estimation
            self.policy = slim.fully_connected(hidden,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            # Only the worker network need ops for loss functions and gradient updating.
            # if scope != 'global':            
            if not match(r'global',scope):
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,
                                                 dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy
                                                         * self.actions_onehot,
                                                         [1])

                # Loss functions
                self.value_loss = tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy
                                               * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5*self.value_loss+self.policy_loss-self.entropy*0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,
                                                               40) # ARA - 1e6;               
                
                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, global_name)
                self.apply_grads = trainer.apply_gradients(zip(grads,
                                                               global_vars))

                

                
