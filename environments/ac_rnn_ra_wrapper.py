# -*- coding: utf-8 -*-

"""Wrappers that provide agents an environment API.

This module contains wrappers that allow an agent to emulate the
interface of an environment object.  With a similar API to environment
objects, agents can be stacked for hierarchical reinforcement learning
with good code incapsulation and minimal adjustment to worker classes.

Todo:
    * Encapsulate code for running agent episodes. It is duplicated in
      the worker classes.
    * Add an internal critic to compute intrinsic rewards.
    * Consider storing extrinsic rewards and performance statistics for
      meta-agent.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import update_target_graph, process_frame, discount
from agents.ac_rnn_ra_network import AC_rnn_ra_Network

class AC_rnn_ra_Wrapper():
    """Wraps an AC agent with rnn support and meta-learning.

    """

    def __init__(self,game,name,s_shape,a_size,trainer,
                 global_episodes, # args to Worker.init(...)
                 # args to Worker.work(...)
                 max_episode_length,update_ival,gamma,lam,
                 model_path):

        self.env = game
        # ARA - todo: generalize this
        self.name = "wrapper_" + str(name)
        self.s_shape = s_shape
        self.a_size = a_size
        # self.trainer = trainer
        self.model_path = model_path
        self.global_episodes = global_episodes
        # self.increment = self.global_episodes.assign_add(1)
        # self.summary_writer = summary_writer  ???
        
        self.max_episode_length = max_episode_length
        self.update_ival = update_ival
        self.gamma = gamma
        self.lam = lam
        self.sess = None
        # self.saver = tf.train.Saver(max_to_keep=5)
        self.im = None

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.total_step_count = 0

        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.local_AC = AC_rnn_ra_Network(s_shape,a_size,self.name,
                                          trainer,hlvl=1)
        # todo: 'global' -> ?
        self.update_local_ops = update_target_graph('global_1',self.name)

        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(name))

        self.flags = {'render':False,
                      'train':True,
                      'verbose':False}

    def train(self,rollout,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [np.array([0]*len(actions[0]))] \
                       + actions[:-1].tolist()
        next_observations = rollout[:,3] # ARA - currently unused
        values = rollout[:,5]
            
        # Here we take the rewards and values from the rollout, and use
        # them to generate the advantage and discounted returns.  The
        # advantage function uses "Generalized Advantage Estimation"
        # Based on: https://github.com/awjuliani/DeepRL-Agents
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,self.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + self.gamma * self.value_plus[1:] \
                     - self.value_plus[:-1]
        advantages = discount(advantages,self.gamma*self.lam)
        
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

        v_l,p_l,e_l,g_n,v_n,_ = self.sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)

        return v_l/len(rollout),p_l/len(rollout),e_l/len(rollout),g_n,v_n

    # Runs after episode completion. Perform a training op. Update graphs.
    def reset(self):
        return self.env.reset()

    # Returns the current environment state
    def render_meta_state(self, g):
        if self.im is None:
            self.im = plt.imshow(self.getState())
            plt.ion()

        s = self.getState()
        m_s = self.get_meta_state(s,self.get_mask(g))
        image = self.visualize_meta_state(m_s)
        self.im.set_data(image)
        plt.pause(0.0001)
        plt.draw()
        return image

    def render(self):
        return self.env.render()

    def getState(self):
        return self.env.getState()

    # compute meta-agent reward
    def meta_reward(self,s,g,sp):
        # meta-reward is sum of green removed
        return np.sum(s[:,:,1]-sp[:,:,1])/(4.0**2)

    # ARA - todo: add to critic and generalize to get "filtered actions"
    def get_mask(self,a,size=(84,84),grid=(4,4),brdr=4):
        """Generate a mask with rectangular cell."""
        ni,nj = grid # divide a mask into a cell regions
        i = np.round(np.linspace(0,size[0]-2*brdr,ni+1)).astype(int)+brdr
        j = np.round(np.linspace(0,size[1]-2*brdr,nj+1)).astype(int)+brdr    
        ri = a // 4 # a should be between 0 and np.prod(grid)
        rj = a % 4
        mask = np.zeros(size)
        mask[i[ri]:i[ri+1],j[rj]:j[rj+1]] = 1.0 # fill grid cell 
        return mask

    # ARA - todo: add to critic and generalize
    def intrinsic_reward(self,s,a,sp,f,g):
        """Intrinsic reward from internal critic in hierarchical RL.

        Arguments:
        s: state/255 prior to taking action
        a: action
        sp: state/255 after taking action
        f: extrinsic reward
        g: current goal
        Returns:
        intrinsic_reward: reward based on agreement with the meta goal
        done: Terminal wrapped_env?
        """
        done = False
        r = -0.05
        # r = 0.0  
        # if the agent's past and present state is inside the masked region
        if f < 0:
            done = True
            r = -10

        #small reward for moving slowly
        if np.sum(s[:,:,2].astype(bool)*sp[:,:,2]) > 0:
            r += 0.05
                
        if np.sum(g.astype(bool)*s[:,:,2]) > 3.5 \
             and np.sum(g.astype(bool)*sp[:,:,2]) > 3.5:
            # f_diff = np.sum(sp[:,:,2]-s[:,:,2])/4
            # r += 0.35*np.exp(-f_diff)
            r += 1
            done = True
            # if r > 0:
            #     done = True
        # elif overlap(s[:,:,2],g) == 1 and overlap(sp[:,:,2],g) == 1:
        #     o = overlap(s[:,:,2], sp[:,:,2])
        #     r += f
        #     if(o > 0 and f == 0):
        #         r += o
        #         done = True            
        i_r = np.clip(r,-1,1)
        # if(self.flags['verbose']):
        #     print('intrinsic reward: ' + str(i_r))
        return i_r, done

    def get_meta_state(self,s,g):
        """compute the 4 channel meta-state from meta-action (goal / option)
        Parameters
        ==========
        s: the raw state
        g: the goal mask
        """
        return np.dstack([process_frame(s),g]) # stack state and goal

    def visualize_meta_state(self,s):
        """
        return a 3-channel state frame for policy visulaization
        Parameters
        ==========
        s: a 4 channel state where the last channel is the meta_goal
        """

        sf = s[:,:,:-1].copy()
        sf[:,:,1] += 0.5*s[:,:,-1]
        sf[:,:,1] /= np.max(sf[:,:,1])
        return sf

    #TODO Pull out tensor flow calculations from step, as a precursor to pulling out model from
    # this env wrapper
    def init_episode():
        self.rnn_state = self.local_AC.state_init
    
    def step(self,m_a):
        """Take a step in this meta_environment
        This single meta_step involves possibly many steps in the environment
        Parameters
        ==========
        m_a: an action of the meta_agent, which is also a goal of this sub agent
           Current this is an input to the get_mask() function
        """


        if self.sess is None: # I cannot init before the sess exists
            self.sess = tf.get_default_session()
            # summary = tf.Summary().add_graph(sess.graph)
            self.summary_writer.add_graph(self.sess.graph)
            # self.summary_writer.flush()

        
            
        self.sess.run(self.update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        i_r = 0
        m_r = 0
        a = np.array([0]*self.a_size)        
        s = self.env.getState() # The meta-agent is responsible for resetting
        g = self.get_mask(m_a)
        s = self.get_meta_state(s,g)
        s0 = s[:,:,:-1].copy()
        episode_frames.append(self.visualize_meta_state(s))        
        rnn_state = self.local_AC.state_init
        
        while d == False:
            # Take an action using probabilities from policy
            # network output.
            feed_dict={self.local_AC.inputs:[s],
                       self.local_AC.prev_actions:[a],
                       self.local_AC.prev_rewards:[[i_r]],
                       self.local_AC.is_training_ph:False,
                       self.local_AC.state_in[0]:rnn_state[0],
                       self.local_AC.state_in[1]:rnn_state[1]}
            a,v,rnn_state = self.sess.run([self.local_AC.sample_a,
                                           self.local_AC.value,
                                           self.local_AC.state_out], 
                                          feed_dict=feed_dict)
                    
            
            s1,f,m_d = self.env.step(a)

            m_r += f
            s1 = self.get_meta_state(s1,g)
            episode_frames.append(self.visualize_meta_state(s1))

            if(self.flags['render']):
                self.render_meta_state(m_a)

            # ARA - todo: make into internal critic or provide a env. wrapper
            i_r,i_d = self.intrinsic_reward(s,a,s1,f,g)

            d = m_d or i_d or episode_step_count == self.max_episode_length-1

                        
            episode_buffer.append([s,a,i_r,s1,d,v[0,0]])
            episode_values.append(v[0,0])

            episode_reward += i_r
            s = s1                    
            episode_step_count += 1
            self.total_step_count += 1
                                        
            # If the episode hasn't ended, but the experience
            # buffer is full, then we make an update step using
            # that experience rollout.
            if len(episode_buffer) == self.update_ival and d != True and \
               episode_step_count != self.max_episode_length - 1 and \
                                     self.flags['train']:
                # Since we don't know what the true final return
                # is, we "bootstrap" from our current value
                # estimation.
                v1 = self.sess.run(self.local_AC.value, 
                    feed_dict={self.local_AC.inputs:[s],
                               self.local_AC.prev_actions:[a],
                               self.local_AC.prev_rewards:[[i_r]],
                               self.local_AC.is_training_ph:False,
                               self.local_AC.state_in[0]:rnn_state[0],
                               self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,v1)
                episode_buffer = []
                self.sess.run(self.update_local_ops)

        if(self.flags['verbose']):
            print('intrisic episode reward: ' + str(episode_reward))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step_count)
        self.episode_mean_values.append(np.mean(episode_values))
                
        # Update the network using the experience buffer at the
        # end of the episode.
        if len(episode_buffer) != 0 and \
           self.flags['train']:
            v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,0.0)
            # ARA - todo: store statistics for summary writer.


            episode_count = self.sess.run(self.global_episodes)

            if episode_count % 50 == 0 and episode_count != 0:

                mean_reward = np.mean(self.episode_rewards[-5:])
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_value = np.mean(self.episode_mean_values[-5:])
                
                summary = tf.Summary()
                summary.value.add(tag='Subagent/Perf/Length',
                                  simple_value=float(mean_length))
                summary.value.add(tag='Subagent/Perf/Intrinsic Reward',
                                  simple_value=float(mean_reward))
                summary.value.add(tag='Subagent/Perf/Value',
                                  simple_value=float(mean_value))
                summary.value.add(tag='Subagent/Perf/Total Step Count',
                                  simple_value=float(self.total_step_count))
                summary.value.add(tag='Subagent/Losses/Value Loss',
                              simple_value=float(v_l))
                summary.value.add(tag='Subagent/Losses/Policy Loss',
                                  simple_value=float(p_l))
                summary.value.add(tag='Subagent/Losses/Entropy',
                                  simple_value=float(e_l))
                summary.value.add(tag='Subagent/Losses/Grad Norm',
                                  simple_value=float(g_n))
                summary.value.add(tag='Subagent/Losses/Var Norm',
                                  simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()
                    

        # ARA - todo: check if max meta-episodes is reached in meta-agent
        #       only send a done (m_d) signal if inner env. needs resetting.
        return s[:,:,:-1],m_r,m_d 
