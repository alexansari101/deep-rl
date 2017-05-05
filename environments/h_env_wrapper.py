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

class H_Env_Wrapper():
    """Wraps an AC agent with rnn support and meta-learning.

    """

    def __init__(self,agent,
                 global_episodes, # args to Worker.init(...)
                 # args to Worker.work(...)
                 max_ep_len,gamma,lam,
                 model_path, grid_size):

        self.env = agent.env
        # ARA - todo: generalize this

        self.model_path = model_path
        self.global_episodes = global_episodes
        # self.increment = self.global_episodes.assign_add(1)

        self.max_ep_len = max_ep_len

        self.gamma = gamma
        self.lam = lam
        self.sess = None
        self.im = None
        self.grid_size = grid_size

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.total_step_count = 0

        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        self.agent = agent
        # todo: 'global' -> ?
        self.update_local_ops = update_target_graph('global_1',agent.name)

        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(agent.name))

        self.flags = {'render':False,
                      'train':True,
                      'verbose':False}
        self.frames = [] #frames for saving movies
        self.last_obs = []


    # Runs after episode completion. Perform a training op. Update graphs.
    def reset(self):
        self.last_obs = self.env.reset()
        self.agent.reset_agent()
        return self.last_obs

    # Returns the current environment state
    def render_meta_state(self, g):
        if self.im is None:
            self.im = plt.imshow(self.get_last_obs())
            plt.ion()

        image = self.get_meta_state_image(g)
        self.im.set_data(image)
        plt.pause(0.0001)
        plt.draw()
        return image

    def get_meta_state_image(self, g):
        s = self.get_last_obs()
        m_s = self.get_meta_state(s,self.get_mask(g))
        return self.visualize_meta_state(m_s)

    def get_frames(self):
        return self.frames
    

    def render(self):
        return self.env.render()

    def get_last_obs(self):
        """Returns the last observation."""
        return self.last_obs

    # compute meta-agent reward
    def meta_reward(self,s,g,sp):
        # meta-reward is sum of green removed
        return np.sum(s[:,:,1]-sp[:,:,1])/(4.0**2)

    # ARA - todo: add to critic and generalize to get "filtered actions"
    def get_mask(self,a,size=(84,84),brdr=4):
        """Generate a mask with rectangular cell."""
        ni,nj = self.grid_size # divide a mask into a cell regions
        i = np.round(np.linspace(0,size[0]-2*brdr,ni+1)).astype(int)+brdr
        j = np.round(np.linspace(0,size[1]-2*brdr,nj+1)).astype(int)+brdr    
        ri = a // ni # a should be between 0 and np.prod(grid)
        rj = a % ni
        mask = np.zeros(size)
        mask[i[ri]:i[ri+1],j[rj]:j[rj+1]] = 1.0 # fill grid cell 
        return mask

    # ARA - todo: add to critic and generalize
    def intrinsic_reward(self,s,a,sp,f,m_d,g):
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

        
        if m_d:
            done = True
            r = f

        #small reward for moving slowly
        if np.sum(s[:,:,2].astype(bool)*sp[:,:,2]) > 0:
            r += 0.05

        #large reward if the agent's past and present
        #  state is inside the masked region
        if np.sum(g.astype(bool)*s[:,:,2]) > 3.5 \
             and np.sum(g.astype(bool)*sp[:,:,2]) > 3.5:
            r += 1
            done = True

        i_r = np.clip(r,-1,1)

        return i_r, done

        #Same Reward as Alex Used
        # done = False
        # r = -0.05
        # # if the agent's past and present state is inside the masked region
        # if f < 0:
        #     done = True
        #     r = -10
        # elif np.sum(g.astype(bool)*s[:,:,2]) > 3.5 \
        #      and np.sum(g.astype(bool)*sp[:,:,2]) > 3.5:
        #     f_diff = np.sum(sp[:,:,2]-s[:,:,2])/4
        #     if(f_diff > 0.0001):
        #         print(f_diff)
        #     r += 0.35*np.exp(-f_diff)
        #     if r > 0:
        #         done = True
        # return r, done    

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
            self.summary_writer.add_graph(self.sess.graph)
        
            
        self.sess.run(self.update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        i_r = 0
        m_r = 0
        s = self.get_last_obs() # The meta-agent is responsible for resetting
        g = self.get_mask(m_a)
        s = self.get_meta_state(s,g)
        s0 = s[:,:,:-1].copy()
        episode_frames.append(self.visualize_meta_state(s))        

        
        while d == False:
            # Take an action using probabilities from policy
            # network output.

            a, v = self.agent.sample_av(s, self.sess, i_r)

            
            s1,f,m_d = self.env.step(a)
            self.last_obs = s1

            m_r += f
            s1 = self.get_meta_state(s1,g)
            episode_frames.append(self.visualize_meta_state(s1))

            if(self.flags['render']):
                self.render_meta_state(m_a)

            # ARA - todo: make into internal critic or provide a env. wrapper
            i_r,i_d = self.intrinsic_reward(s,a,s1,f,m_d,g)

            d = m_d or i_d or episode_step_count == self.max_ep_len-1

                        
            episode_buffer.append([s,a,i_r,s1,d,v[0,0]])
            episode_values.append(v[0,0])

            episode_reward += i_r
            s = s1                    
            episode_step_count += 1
            self.total_step_count += 1
                                        

        if(self.flags['verbose']):
            print('intrisic episode reward: ' + str(episode_reward))

                
        # Update the network using the experience buffer at the
        # end of the episode.
        if len(episode_buffer) != 0 and \
           self.flags['train']:
            v_l,p_l,e_l,g_n,v_n = self.agent.train(episode_buffer,
                                                   self.sess,
                                                   self.gamma, self.lam, 0.0)

            episode_count = self.sess.run(self.global_episodes)

            if episode_count % 50 == 0:

                summary = tf.Summary()
                summary.value.add(tag='Subagent/Perf/Length',
                                  simple_value=float(episode_step_count))
                summary.value.add(tag='Subagent/Perf/Intrinsic Reward',
                                  simple_value=float(episode_reward))
                summary.value.add(tag='Subagent/Perf/Value',
                                  simple_value=float(np.mean(episode_values)))
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
        self.frames = episode_frames
        return s[:,:,:-1],m_r,m_d 
