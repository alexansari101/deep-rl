# -*- coding: utf-8 -*-

"""Actor-critic worker interface

This module contains a worker agent that uses actor-critic networks.  By
threading the workers work() function they can be run asynchronously as
in A3C.  Each worker contains a copy of the AC network that it syncs
with a global network as in A3C.  The network specifies the policy the
worker uses to send actions to its copy of the environment.  The worker
stores the resulting rewards, observations, and any other variables
required for training the AC network.  The worker continuously loops
through through episodes and computes the graidents required to train
its local copy of the AC network, then updates the global network before
it re-syncs its local copy.  A worker can save summary statistics and
can visualize policy performance, e.g., by making animations, at desired
intervals.

Example:
    Using tensorboard to display stats during training:
    $ tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',\
      worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',\
      worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7'

Todo:
    * Develop a common worker interface. Consider an abstract base class
      with multiple derived classes to add specialized features.
    * Allow the worker class to accept an AC network as an argument.
    * Make summary writer and generation of multimedia separate classes
      (or functions).
    * Find a way to generate multimedia that works with Threading. Tk
      needs to be run in the main thread.
    * Make the summary_writer save in the model_path folder.
    * See if the Worker really needs a copy of the global AC network.
    * Avoid using hard-coded name 'global' for global AC network
    * Implement a fixed-length buffer to store episode stats in worker.
    * Consider incorporating the sess.run(..) commands required for training
      and updates into the ac networks classes since they are specific to each
      network.
    * Encapsulate code for running agent episodes. It is duplicated in
      the agent wrapper environment classes.

"""

import numpy as np
import tensorflow as tf

from util import update_target_graph, process_frame, discount
import matplotlib

matplotlib.use('PS')
#PS can run on parallel threads, TK (default) cannot
#Probably this command should be moved somewhere else

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import time
from datetime import timedelta

from agents.ac_rnn_ra_network import AC_rnn_ra_Network

class AC_Agent_Base():
    """Advantage actor-critic worker Interface.

    This implementation includes inputs for an rnn, and additional
    rnn inputs of previous rewards and actions for meta-learning.
    
    """
    
    def __init__(self,game,name,s_shape,a_size,trainer,model_path,
                 global_episodes, hlvl=0):
        """Initialize the worker environment, AC net, and trainer.

        Args:
            game: An environment object
            name (str): name of the worker agent.
            s_shape (list): shape of received environment states (observations)
            a_size (int): the dimension of the continuous action vector.
            trainer: a tensorflow optimizer from the tf.train module.
            model_path: folder under which to save the model
            global_episodes: a tensorflow tensor to store the global
                episode count
        
        """
        self.name = name
        self.s_shape = s_shape
        self.a_size = a_size
        self.name = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.movie_path = model_path + "/movies_"
        self.is_writer = name.endswith('0')
        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(self.name))

        self.hlvl = hlvl
        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.update_local_ops = update_target_graph('global_'+str(hlvl),self.name)  

        self.env = game
        self.prev_a = None

        self.local_AC = None #Must be defined in inherited class


    def train(self,rollout,sess,gamma,lam,bootstrap_value):
        raise Exception("NotImplementedException")

    def sample_av(self, s, sess, prev_r):
        raise Exception("NotImplementedException")

    def reset_agent(self):
        raise Exception("NotImplementedException")

    def start_trial(self):
        raise Exception("NotImplementedException")
        
    def work(self,max_episode_length,update_ival,gamma,lam,global_AC,sess,
             coord,saver):
        raise Exception("NotImplementedException")

    def write_summary(self, data_dict, ep_count):
        """Writes summaries to be viewed in tensorboard
        data_dict is a dictionary of {name:number} pairs"""
        summary = tf.Summary()
        prefix = 'agent_lvl_' + str(self.hlvl) + '/'
        for name in data_dict.keys():
            summary.value.add(tag=prefix+name,
                              simple_value=float(data_dict[name]))
        self.summary_writer.add_summary(summary, ep_count)
        self.summary_writer.flush()


    def evaluate(self, sess, n=0):
        episode_count = sess.run(self.global_episodes)
        s = self.env.reset()
        self.reset_agent()
        self.start_trial()

        s = process_frame(s)
        d = False
        r = 0
        episode_r = 0

        is_meta = hasattr(self.env, 'flags')

        if is_meta:
            self.env.flags['train'] = False
            self.env.flags['verbose'] = True

        printing = True

        frames = []
        
        while d == False:
            a, v = self.sample_av(s, sess, r)
                
            s1,r,d = self.env.step(a)

            if is_meta:
                frames += self.env.get_frames()
            else:
                frames.append(s1)
                
            episode_r += r
            s = process_frame(s1)
        print('episode reward: ' + str(episode_r))
        
        if not printing:
            return

        fig = plt.figure()

        l = plt.imshow(frames[0])

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        movie_path = self.movie_path + "episode_" + str(n) + ".mp4"
        with writer.saving(fig, movie_path, 100):
            for f in frames:
                l.set_data(f)
                writer.grab_frame()
        plt.close()

        if is_meta:
            self.env.flags['train'] = True
            self.env.flags['verbose'] = False