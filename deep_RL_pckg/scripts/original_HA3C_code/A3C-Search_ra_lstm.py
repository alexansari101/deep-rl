#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A3C Rewards-Driven Search.

This module implements a modified A3C agent that includes continuous
policies, additional normalization/regularization, and an RNN layer that
accepts addition features, e.g., rewards and actions, for meta-learning.

Example:
    Using tensorboard to display stats during training:
    $ tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',\
      worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',\
      worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7'

Todo:
    * Add input arguments and a main function.
    * Make abstact class to handle threading.
    * Fix agents __init__.py to allow 'from agents import AC_Worker ...'
"""

import os
import numpy as np
import tensorflow as tf
import threading
import multiprocessing

from environments.hregion_search import gameEnv
from agents.ac_rnn_ra_network import AC_rnn_ra_Network
from agents.ac_rnn_ra_worker import AC_rnn_ra_Worker


###################
# Hyperparameters #
###################
max_episode_length = 50
update_ival = np.inf    # train after this many steps if < max_episode_length
gamma = .99             # discount rate for reward discounting
lam = 1                 # .97; discount rate for advantage estimation
s_shape = [84,84,3]     # Observations are rgb frames 84 x 84
a_size = 2              # Agent can move in a line
gameArgs = {}
load_model = False
model_path = './model'


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5) # beta1=0.99
    master_network = AC_rnn_ra_Network(s_shape,a_size,
                                       'global_0',None) # Global network
    num_workers = multiprocessing.cpu_count() # number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(AC_rnn_ra_Worker(gameEnv(**gameArgs),i,s_shape,a_size,
                                        trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,update_ival,gamma,
                                          lam,master_network,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

