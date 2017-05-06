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
from util import update_target_graph

class H_Env_Wrapper():
    """Wraps an AC agent with rnn support and meta-learning.

    """

    def __init__(self,agent,
                 subgoal,
                 global_episodes, # args to Worker.init(...)
                 # args to Worker.work(...)
                 max_ep_len,gamma,lam,
                 model_path):

        self.env = agent.env
        self.subgoal = subgoal
        # ARA - todo: generalize this

        self.model_path = model_path
        self.global_episodes = global_episodes
        # self.increment = self.global_episodes.assign_add(1)

        self.max_ep_len = max_ep_len

        self.gamma = gamma
        self.lam = lam
        self.sess = None
        self.im = None

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.total_step_count = 0
        self.episode_count = 0

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


    def get_frames(self):
        return self.frames
    
    def render(self):
        return self.env.render()

    def get_last_obs(self):
        """Returns the last observation."""
        return self.last_obs



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
        s0 = s.copy()
        self.subgoal.set_meta_action(m_a)
        s = self.subgoal.augment_obs(s)

        episode_frames.append(self.subgoal.visualize(s))
        self.agent.start_trial()
        
        while d == False:
            # Take an action using probabilities from policy
            # network output.

            a, v = self.agent.sample_av(s, self.sess, i_r)

            s1,f,m_d = self.env.step(a)
            self.last_obs = s1
            s1 = self.subgoal.augment_obs(s1)

            m_r += f

            episode_frames.append(self.subgoal.visualize(s1))

            if(self.flags['render']):
                self.render_meta_state(m_a)

            # ARA - todo: make into internal critic or provide a env. wrapper
            i_r,i_d = self.subgoal.intrinsic_reward(s,a,s1,f,m_d)
            # if(self.flags['verbose']):
            #     print('i_r: ' + str(i_r))
            s = s1

            d = m_d or i_d or episode_step_count == self.max_ep_len-1

                        
            episode_buffer.append([s,a,i_r,s1,d,v[0,0]])
            episode_values.append(v[0,0])
            episode_reward += i_r
            episode_step_count += 1
            self.total_step_count += 1
                                        
        self.episode_count += 1
        if(self.flags['verbose']):
            print('total intrisic episode reward: ' + str(episode_reward))
            print('subagent length: ' + str(episode_step_count))

                
        # Update the network using the experience buffer at the
        # end of the episode.
        if len(episode_buffer) != 0 and \
           self.flags['train']:
            v_l,p_l,e_l,g_n,v_n = self.agent.train(episode_buffer,
                                                   self.sess,
                                                   self.gamma, self.lam, 0.0)



            if self.episode_count % 50 == 0:
                global_ep_count = self.sess.run(self.global_episodes)

                summary = tf.Summary()
                summary.value.add(tag='Subagent/Perf/Length',
                                  simple_value=float(episode_step_count))
                summary.value.add(tag='Subagent/Perf/Intrinsic Reward',
                                  simple_value=float(episode_reward))
                summary.value.add(tag='Subagent/Perf/Value',
                                  simple_value=float(np.mean(episode_values)))
                summary.value.add(tag='Subagent/Perf/Total Step Count',
                                  simple_value=float(self.total_step_count))
                summary.value.add(tag='Subagent/Perf/Global Episode Count',
                                  simple_value=float(global_ep_count))
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
                self.summary_writer.add_summary(summary, self.episode_count)
                self.summary_writer.flush()
                    

        # ARA - todo: check if max meta-episodes is reached in meta-agent
        #       only send a done (m_d) signal if inner env. needs resetting.
        self.frames = episode_frames
        return s[:,:,:-1],m_r,m_d 
