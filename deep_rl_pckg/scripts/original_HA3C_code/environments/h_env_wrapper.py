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
from util import update_target_graph, process_frame
from IPython import embed

class H_Env_Wrapper():
    """Wraps an AC agent with rnn support and meta-learning.

    """

    def __init__(self, sess, agent, lp, model_path):
        """Created an intermediate level environment in the hierarchy
        
        Args:
            agent:  the lower level agent (currently must be ac agent)
            subgoal: the subgoal containing the intrinsic rewards
            global_episodes: tensor to store global ep count
            lp: learning parameters
            model_path: save folder
        """

        self.env = agent.env
        self.subgoal = agent.env
        # ARA - todo: generalize this

        self.observation_space = self.env.m_observation_space
        self.action_space = self.env.m_action_space

        self.model_path = model_path
        self.global_episodes = agent.global_episodes
        # self.increment = self.global_episodes.assign_add(1)


        self.lam         = lp['lambda']
        self.max_ep_len  = lp['max_episode_length']
        self.update_ival = lp['update_ival']
        self.gamma       = lp['gamma']

        self.sess = sess
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


        self.summary_writer = tf.summary.FileWriter(model_path + "/train_"
                                                    + str(agent.name))

        self.flags = {'render':False,
                      'train':True,
                      'verbose':False}
        
        self.frames = [] #frames for saving movies
        self.poses = [] #poses of the hero in each frame
        self.last_obs = []
        self.last_pose = []

    # Runs after episode completion. Perform a training op. Update graphs.
    def reset(self,pose_init):
        self.last_obs = self.env.reset(pose_init)
        self.agent.reset_agent()
        return self.last_obs


    def get_frames(self):
        return self.frames
    
    def get_poses(self):
        return self.poses

    def render(self):
        """Should render the env, currently not working"""
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
            # embed()
            self.sess = tf.get_default_session()
            self.summary_writer.add_graph(self.sess.graph)
        
            
        self.sess.run(self.agent.update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_poses = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        i_r = 0
        m_r = 0
        s = self.get_last_obs() # The meta-agent is responsible for resetting
        s = process_frame(s)

        self.subgoal.set_meta_action(m_a)
        s = self.subgoal.augment_obs(s)
        # episode_frames.append((self.subgoal.visualize(s),['i_r  =  0', 'm_r  =  0', 'step = 0']))   
        # episode_poses.append(self.last_pose)


        self.agent.start_trial()        
        while d == False:
            # Take an action using probabilities from policy
            # network output.
            a,v = self.agent.sample_av(s, self.sess, i_r)
            s1,f,m_d = self.env.step(a)
            pose = self.env.env.getHeroPosition()
            # print(pose)
            self.last_obs = s1.copy()
            s1 = process_frame(s1)
            s1 = self.subgoal.augment_obs(s1)



            # ARA - todo: make into internal critic or provide a env. wrapper
            i_r, m_r_step, i_d = self.subgoal.intrinsic_reward(s,a,s1,f,m_d)
            m_r += m_r_step
            # if(self.flags['verbose']):
            #     print('i_r: ' + str(i_r))


            d = m_d or i_d or episode_step_count == self.max_ep_len-1

            data = ['i_r  = ' + str(i_r),
                    'm_r  = ' + str(m_r_step),
                    'd    = ' + str(d),
                    'step = ' + str(episode_step_count),
                    'a    = ' + str(np.round(a,2)),
                    'm_a  = ' + str(np.round(m_a,2)),
                    'v    = ' + str(v[0,0])]
                    
            episode_frames.append((self.subgoal.visualize(s1), data))
            # print(pose[:2])
            episode_poses.append(pose)
            episode_buffer.append([s,a,i_r,s1,d,v[0,0]])
            episode_values.append(v[0,0])
            episode_reward += i_r
            s = s1
            episode_step_count += 1
            self.total_step_count += 1
        
        self.episode_count += 1
        if(self.flags['verbose']):
            print('\ttotal intrisic episode reward: ' + str(episode_reward))
            print('\tsubagent length: ' + str(episode_step_count))

                
        # Update the network using the experience buffer at the
        # end of the episode.
        if len(episode_buffer) != 0 and \
           self.flags['train']:
            v_l,p_l,e_l,g_n,v_n = self.agent.train(episode_buffer,
                                                   self.sess,
                                                   self.gamma, self.lam, 0.0)



            if self.episode_count % 50 == 0:
                global_ep_count = self.sess.run(self.global_episodes)

                data = {'Perf/Intrinsic Reward' : episode_reward,
                        'Perf/Length'           : episode_step_count,
                        'Perf/Value'            : np.mean(episode_values),
                        'Perf/Total Step Count' : self.total_step_count,
                        'Perf/Global Ep Count'  : global_ep_count,
                        'Losses/Value Loss'     : v_l,
                        'Losses/Policy Loss'    : p_l,
                        'Losses/Entropy'        : e_l,
                        'Losses/Grad Norm'      : g_n,
                        'Losses/Var Norm'       : v_n}

                self.agent.write_summary(data, self.episode_count)
                    

        # ARA - todo: check if max meta-episodes is reached in meta-agent
        #       only send a done (m_d) signal if inner env. needs resetting.
        self.frames = episode_frames
        # print(episode_poses)
        self.poses  = episode_poses
        return self.last_obs, m_r, m_d 
