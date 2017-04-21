# -*- coding: utf-8 -*-

"""Actor-critic workers

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
    * See AC_rnn_ra_Worker() todo list

"""

import numpy as np
import tensorflow as tf
import time
from datetime import timedelta

from util import update_target_graph, process_frame, discount
from agents.ac_network import AC_Network

class AC_Worker():
    """Simple advantage actor-critic worker with discrete actions.
    
    """
    
    def __init__(self,game,name,s_shape,a_size,trainer,model_path,
                 global_episodes):
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
        self.name = "worker_" + str(name)
        self.s_shape = s_shape
        self.a_size = a_size
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(self.number))

        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.local_AC = AC_Network(s_shape,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global_0',self.name)  

        self.env = game

    def train(self,global_AC,rollout,sess,gamma,lam,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3] # ARA - currently unused
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use
        # them to generate the advantage and discounted returns.  The
        # advantage function uses "Generalized Advantage Estimation"
        # Based on: https://github.com/awjuliani/DeepRL-Agents
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:]-self.value_plus[:-1]
        advantages = discount(advantages,gamma*lam)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     # ARA - using np.stack to support Ndarray states
                     self.local_AC.inputs:np.stack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
                                          self.local_AC.policy_loss,
                                          self.local_AC.entropy,
                                          self.local_AC.grad_norms,
                                          self.local_AC.var_norms,
                                          self.local_AC.apply_grads],
                                         feed_dict=feed_dict)
        return v_l/len(rollout),p_l/len(rollout),e_l/len(rollout),g_n,v_n

    def evaluate(self, sess):
        episode_count = sess.run(self.global_episodes)
        s = self.env.reset()
        s = process_frame(s)
        d = False
        r = 0
        episode_r = 0

        self.env.flags['render'] = True
        self.env.flags['train'] = False
        self.env.flags['verbose'] = True
        
        while d == False:
            a_dist,v = sess.run([self.local_AC.policy,
                                 self.local_AC.value], 
                                feed_dict={self.local_AC.inputs:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            s1,r,d = self.env.step(a)
            episode_r += r
        print('episode reward: ' + str(episode_r))
            
        
    def work(self,max_episode_length,update_ival,gamma,lam,global_AC,sess,
             coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        t0 = time.time()
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                state_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)

                while d == False:
                    # Take an action using probabilities from policy
                    # network output.
                    a_dist,v = sess.run([self.local_AC.policy,
                                        self.local_AC.value], 
                                        feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    s1,r,d = self.env.step(a)
#                     if episode_count == 50:
#                         coord.request_stop()
                    episode_frames.append(s1)
                    s1 = process_frame(s1)
                    if episode_step_count == max_episode_length-1:
                        d = True
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    state_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                                        
                    # If the episode hasn't ended, but the experience
                    # buffer is full, then we make an update step using
                    # that experience rollout.
                    if len(episode_buffer) == update_ival and d != True and \
                       episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return
                        # is, we "bootstrap" from our current value
                        # estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(global_AC,
                                                         episode_buffer,
                                                         sess,gamma,lam,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                
                # Update the network using the experience buffer at the
                # end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(global_AC,episode_buffer,
                                                     sess,gamma,lam,0.0)
                if episode_count % 1000 == 0 and self.name == 'worker_0':
                    saver.save(sess,self.model_path+'/model.ckpt', episode_count)
                    
                    s_dt = str(timedelta(seconds=time.time()-t0))
                    print("Saved Model " + str(episode_count) + '\tat time ' + s_dt)

                    
                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:

                    mean_state_value = np.mean(state_values)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward',
                                      simple_value=float(episode_reward))
                    summary.value.add(tag='Perf/Episode Length',
                                      simple_value=float(episode_step_count))
                    summary.value.add(tag='Perf/Mean State Value',
                                      simple_value=float(mean_state_value))
                    summary.value.add(tag='Perf/Global Episodes',
                                      simple_value=float(sess.run(self.global_episodes)))
                    summary.value.add(tag='Losses/Value Loss',
                                      simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss',
                                      simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy',
                                      simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm',
                                      simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm',
                                      simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
