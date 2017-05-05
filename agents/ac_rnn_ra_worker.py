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
from agents.ac_rnn_ra_network import AC_rnn_ra_Network

class AC_rnn_ra_Worker():
    """Advantage actor-critic worker with rnn support and meta-learning.

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
        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(self.name))

        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.local_AC = AC_rnn_ra_Network(s_shape,a_size,self.name,trainer,hlvl)
        self.update_local_ops = update_target_graph('global_'+str(hlvl),self.name)  

        self.env = game

        self.rnn_state = None
        self.prev_a = None


    def train(self,rollout,sess,gamma,lam,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [np.array([0]*self.a_size)] + actions[:-1].tolist()
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
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     # ARA - using np.stack to support Ndarray states
                     self.local_AC.inputs:np.stack(observations),
                     self.local_AC.prev_actions:np.vstack(prev_actions),
                     self.local_AC.prev_rewards:np.vstack(prev_rewards),
                     self.local_AC.is_training_ph:True,
                     self.local_AC.actions:np.vstack(actions),
                     self.local_AC.advantages:advantages,
                     self.local_AC.state_in[0]:rnn_state[0],
                     self.local_AC.state_in[1]:rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
                                          self.local_AC.policy_loss,
                                          self.local_AC.entropy,
                                          self.local_AC.grad_norms,
                                          self.local_AC.var_norms,
                                          self.local_AC.apply_grads],
                                         feed_dict=feed_dict)
        return v_l/len(rollout),p_l/len(rollout),e_l/len(rollout),g_n,v_n

    def sample_av(self, s, sess, prev_r):
        """Returns an action sampled from the agent's policy, and value"""
        # feed_dict={self.local_AC.inputs:[s]}
        feed_dict={self.local_AC.inputs:[s],
                   self.local_AC.prev_actions:[self.prev_a],
                   self.local_AC.prev_rewards:[[prev_r]],
                   self.local_AC.is_training_ph:False,
                   self.local_AC.state_in[0]:self.rnn_state[0],
                   self.local_AC.state_in[1]:self.rnn_state[1]}


        a, v, rnn_state = sess.run([self.local_AC.sample_a,
                                    self.local_AC.value,
                                    self.local_AC.state_out],
                                   feed_dict=feed_dict)
        self.rnn_state = rnn_state
        self.prev_a = a
        return a, v

    def reset_agent(self):
        self.rnn_state = self.local_AC.state_init
        self.prev_a = np.array([0]*self.a_size)
                
        
    def work(self,max_episode_length,update_ival,gamma,lam,global_AC,sess,
             coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = np.array([0]*self.a_size)
                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init

                while d == False:
                    # Take an action using probabilities from policy
                    # network output.
                    feed_dict={self.local_AC.inputs:[s],
                               self.local_AC.prev_actions:[a],
                               self.local_AC.prev_rewards:[[r]],
                               self.local_AC.is_training_ph:False,
                               self.local_AC.state_in[0]:rnn_state[0],
                               self.local_AC.state_in[1]:rnn_state[1]}
                    a,v,rnn_state = sess.run([self.local_AC.sample_a,
                                              self.local_AC.value,
                                              self.local_AC.state_out], 
                                             feed_dict=feed_dict)
                    
                    s1,r,d = self.env.step(a)
#                     if episode_count == 50:
#                         coord.request_stop()
                    episode_frames.append(s1)
                    s1 = process_frame(s1)
                    if episode_step_count == max_episode_length-1:
                        d = True
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

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
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.prev_actions:[a],
                            self.local_AC.prev_rewards:[[r]],
                            self.local_AC.is_training_ph:False,
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0] 
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,
                                                         sess,gamma,lam,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the
                # end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(global_AC,episode_buffer,
                                                     sess,gamma,lam,0.0)
                    
                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'
                                   +str(episode_count)+'.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward',
                                      simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length',
                                      simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value',
                                      simple_value=float(mean_value))
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
