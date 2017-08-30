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
from IPython import embed
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

import rospy
from std_msgs.msg import String, Float32MultiArray

class AC_Agent_Base():
    """Advantage actor-critic worker Interface.

    This implementation includes inputs for an rnn, and additional
    rnn inputs of previous rewards and actions for meta-learning.
    
    """
    
    def __init__(self,env,name,trainer,model_path,
                 lp, hlvl):
        """Initialize the worker environment, AC net, and trainer.

        Args:
            env: An environment object
            name (str): name of the worker agent.
            trainer: a tensorflow optimizer from the tf.train module.
            model_path: folder under which to save the model
            lp: dictionary of parameters related to training
            hlvl: hierarchy level (0 for highest lvl agent)
        """
        self.name = name
        
        self.model_path = model_path
        self.trainer = trainer

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.movie_path = model_path + "/movies_"
        self.is_writer = name.endswith('0')
        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(self.name))

        #learning params
        self.lam         = lp['lambda']
        self.max_ep      = lp['max_episode_length']
        self.update_ival = lp['update_ival']
        self.gamma       = lp['gamma']

        # Share the "global_episodes" variable between networks
        g = [v for v in tf.global_variables() if v.name == 'global_episodes']
        if not g:
            g = [tf.Variable(0,dtype=tf.int32,name='global_episodes',
                            trainable=False)]
        self.global_episodes = g[0]
        self.increment = self.global_episodes.assign_add(1)
        

        self.hlvl = hlvl
        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.update_local_ops = None #Must be defined after local_AC, so the variables exist in tf
        
        self.env = env
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
        
    def work(self, sess, coord, saver):
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

        step = 0
        
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
        
        while d == False and step < self.max_ep:
            a, v = self.sample_av(s, sess, r)
                
            s1,r,d = self.env.step(a)
            episode_r += r
            s = process_frame(s1)
            step += 1


            if is_meta:
                frames += self.env.get_frames()
            else:
                data = ['r = ' + str(r),
                        'd = ' + str(d),
                        'v = ' + str(v),
                        'a = ' + str(a),
                        'step = ' + str(step),
                        'cum_r = ' + str(episode_r)]
                frames.append((s1, data))
                

        print('episode reward: ' + str(episode_r))
        
        if not printing:
            return

        fig = plt.figure()
        f, d = frames[0]
        lf_sp = fig.add_subplot(121)
        l = plt.imshow(f)
        data_plot = fig.add_subplot(122)


        plt.imshow(np.ones(f.shape))
        plt.axis('off')

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Episode '+str(n), artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        movie_path = self.movie_path + "episode_" + str(n) + ".mp4"
        with writer.saving(fig, movie_path, 100):
            for f, data in frames:
                l.set_data(f)
                
                data_plot.cla()
                data_plot.axis('off')

                h = 3
                for text in data:
                    data_plot.text(1,h, text)
                    h+=8
                
                writer.grab_frame()
        plt.close()

        if is_meta:
            self.env.flags['train'] = True
            self.env.flags['verbose'] = False


    def test(self, sess, aqFunction, pose_init, n=0):

        # pose_pub = rospy.Publisher('pose', Float32MultiArray, queue_size=1)        
        print('............created_pose_publisher..................')

        episode_count = sess.run(self.global_episodes)
        # embed()
        s = self.env.reset(pose_init, aqFunction)
        self.reset_agent()
        self.start_trial()

        step = 0
        
        s = process_frame(s)
        d = False
        r = 0
        episode_r = 0

        is_meta = hasattr(self.env, 'flags')

        if is_meta:
            self.env.flags['train'] = False
            self.env.flags['verbose'] = True

        printing = False

        frames = []
        poses = []
        
        # embed()
        while d == False and step < 1:
            print("step ---- " + str(step))
            a, v = self.sample_av(s, sess, r)
        
            s1,r,d = self.env.step(a)
            episode_r += r
            s = process_frame(s1)
            step += 1

            # embed()
            # plt.ion()
            if is_meta:
                current_frame = self.env.get_frames()
                current_pose = self.env.get_poses()
                frames += current_frame
                poses += current_pose
                pose_msg=Float32MultiArray(data=np.array(current_pose).flatten())
                # pose_pub.publish(pose_msg)                 
                print(aqFunction)
                # msg = rospy.wait_for_message("matrix", Float32MultiArray)  
                # print(msg)

            
    

            else:
                data = ['r = ' + str(r),
                        'd = ' + str(d),
                        'v = ' + str(v),
                        'a = ' + str(a),
                        'step = ' + str(step),
                        'cum_r = ' + str(episode_r)]
                frames.append((s1, data))

        plt.figure(1)
        l=plt.imshow(frames[0][0])                    
        
        # plt.figure(2)
        # plt.axis([0, 90, 0, 90])
        
        print('episode reward: ' + str(episode_r))
        
        for ((f, data),(y,x)) in zip(frames, poses):
            plt.figure(1)
            l.set_data(f)
            plt.pause(0.00001)
            
            # plt.figure(2)
            # plt.scatter(x,-y+90)
            # plt.pause(0.00001)
        

        # pose_msg=Float32MultiArray(data=np.array(poses).flatten())
        # pose_pub.publish(pose_msg)

        # plt.show()    
        





        if not printing:
            return poses

        fig = plt.figure()
        f, d = frames[0]
        lf_sp = fig.add_subplot(121)
        l = plt.imshow(f)
        data_plot = fig.add_subplot(122)


        plt.imshow(np.ones(f.shape))
        plt.axis('off')

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Episode '+str(n), artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        movie_path = self.movie_path + "episode_" + str(n) + ".mp4"
        with writer.saving(fig, movie_path, 100):
            for f, data in frames:
                l.set_data(f)
                
                data_plot.cla()
                data_plot.axis('off')

                h = 3
                for text in data:
                    data_plot.text(1,h, text)
                    h+=8
                writer.grab_frame()
        plt.close()

        if is_meta:
            self.env.flags['train'] = True
            self.env.flags['verbose'] = False

