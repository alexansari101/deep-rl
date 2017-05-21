import tensorflow as tf
import numpy as np

from .ac_network import AC_Network
from .ac_rnn_ra_network import AC_rnn_ra_Network
from .ac_worker import AC_Worker
from .ac_rnn_ra_worker import AC_rnn_ra_Worker

import sys
sys.path.append('..')
from environments.h_env_wrapper import H_Env_Wrapper
from intrinsics.grid_goal import GridGoal
from intrinsics.dummy_subgoal import DummyGoal

def get_2lvl_HA3C(env_gen, num_workers, out_folder,
                  grid_size = (4,4)):

    """Returns a hierarchical agent
    lvl 1 is AC
    lvl 2 is AC_RNN_RA"""
    with tf.device("/cpu:0"):
        m_lp = {'lambda'            : 1,
                'gamma'             : .99,
                'update_ival'       : np.inf,
                'max_episode_length': 20}
        lp = {'lambda'            : 1,
              'gamma'             : .99,
              'update_ival'       : np.inf,
              'max_episode_length': 20}
        
        m_trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                          trainable=False)
        env = env_gen()
        m_s_shape = env.observation_space.shape
        subgoal = GridGoal(m_s_shape, grid_size)
        s_shape = subgoal.observation_space.shape
        a_size = env.action_space.n
        m_a_size = subgoal.action_space.n
            
        m_master_network = AC_Network(m_s_shape,m_a_size,'global_0',None) # meta network
        master_network = AC_rnn_ra_Network(s_shape,a_size,'global_1',None)

        workers = []
        
        for i in range(num_workers):
            env = env_gen()

            subgoal = GridGoal(m_s_shape, grid_size)
            agent_1 = AC_rnn_ra_Worker(env, 'agent_1_'+str(i), s_shape, a_size,
                                       trainer, out_folder, global_episodes, lp,
                                       hlvl=1)
            env_1 = H_Env_Wrapper(agent_1, subgoal, global_episodes,
                                  lp, model_path=out_folder)
            # env_1.flags['verbose']=True
            
            agent_0 = AC_Worker(env_1, 'agent_0_'+str(i), m_s_shape, m_a_size, m_trainer,
                                out_folder, global_episodes, m_lp)

            workers.append(agent_0)

        return workers

    
def get_dummy_2lvl_HA3C(env_gen, num_workers, out_folder,
                        grid_size = (4,4)):

    """Returns a hierarchical agent
    lvl 1 is AC
    lvl 2 is AC_RNN_RA"""
    with tf.device("/cpu:0"):
        m_lp = {'lambda'            : 1,
                'gamma'             : .99,
                'update_ival'       : np.inf,
                'max_episode_length': 1}
        lp = {'lambda'            : 1,
              'gamma'             : .99,
              'update_ival'       : np.inf,
              'max_episode_length': 20}

        m_trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                          trainable=False)
        env = env_gen()
        m_s_shape = env.observation_space.shape
        subgoal = DummyGoal(m_s_shape, grid_size)
        s_shape = subgoal.observation_space.shape
        a_size = env.action_space.n
        m_a_size = subgoal.action_space.n
            
        m_master_network = AC_Network(m_s_shape,m_a_size,'global_0',None) # meta network
        master_network = AC_rnn_ra_Network(s_shape,a_size,'global_1',None)

        workers = []
        
        for i in range(num_workers):
            env = env_gen()

            subgoal = DummyGoal(m_s_shape, grid_size)
            agent_1 = AC_rnn_ra_Worker(env, 'agent_1_'+str(i), s_shape, a_size,
                                       trainer, out_folder, global_episodes, lp,
                                       hlvl=1)
            env_1 = H_Env_Wrapper(agent_1, subgoal, global_episodes,
                                  lp, model_path=out_folder)
            # env_1.flags['verbose']=True
            
            agent_0 = AC_Worker(env_1, 'agent_0_'+str(i), m_s_shape, m_a_size, m_trainer,
                                out_folder, global_episodes, m_lp)

            workers.append(agent_0)

        return workers

def get_1lvl_ac_rnn(env_gen, num_workers, out_folder):
    with tf.device("/cpu:0"):
        lp = {'lambda'            : 1,
              'gamma'             : .99,
              'update_ival'       : np.inf,
              'max_episode_length': 20}

        trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                          trainable=False)
        env = env_gen()
        s_shape = env.observation_space.shape
        a_size = env.action_space.n
            
        master_network = AC_rnn_ra_Network(s_shape,a_size,'global_0',None)

        workers = []
        
        for i in range(num_workers):
            env = env_gen()
            workers.append(AC_rnn_ra_Worker(env, 'agent_' + str(i),
                                            s_shape, a_size, trainer,
                                            out_folder, global_episodes, lp))

        return workers


