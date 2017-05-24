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
from intrinsics.rect_goal import RectGoal

def get_2lvl_rnn_ra_x2(env_gen, num_workers, out_folder):
    """Returns a group hierarchical agent workers
    lvl 0 is AC_RNN_RA
    lvl 1 is AC_RNN_RA"""
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

        workers = []
        
        for i in range(num_workers):
            env_1 = RectGoal(env_gen())
            agent_1 = AC_rnn_ra_Worker(env_1, 'agent_1_'+str(i),
                                       trainer, out_folder, lp,
                                       hlvl=1)
            env_0 = H_Env_Wrapper(agent_1, lp, model_path=out_folder)
            
            agent_0 = AC_rnn_ra_Worker(env_0, 'agent_0_'+str(i), m_trainer,
                                       out_folder, m_lp)

            workers.append(agent_0)

        return workers


def get_2lvl_HA3C(env_gen, num_workers, out_folder,
                  grid_size = (4,4)):
    """Returns a group hierarchical agent workers
    lvl 0 is AC
    lvl 1 is AC_RNN_RA"""
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

        workers = []
        
        for i in range(num_workers):
            env_1 = GridGoal(env_gen(), grid_size)
            agent_1 = AC_rnn_ra_Worker(env_1, 'agent_1_'+str(i),
                                       trainer, out_folder, lp,
                                       hlvl=1)
            env_0 = H_Env_Wrapper(agent_1, lp, model_path=out_folder)
            
            agent_0 = AC_Worker(env_0, 'agent_0_'+str(i), m_trainer,
                                out_folder, m_lp)

            workers.append(agent_0)

        return workers

    
def get_dummy_2lvl_HA3C(env_gen, num_workers, out_folder):
    """Returns a group of hierarchical agent workers
    lvl 0 is AC
    lvl 1 is AC_RNN_RA
    Dummy subgoal is used, so the AC does nothing"""
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

        workers = []
        
        for i in range(num_workers):
            env_1 = DummyGoal(env_gen())
            agent_1 = AC_rnn_ra_Worker(env_1, 'agent_1_'+str(i), trainer,
                                       out_folder, lp, hlvl=1)
            env_0= H_Env_Wrapper(agent_1, lp, model_path=out_folder)
            agent_0 = AC_Worker(env_0, 'agent_0_'+str(i), m_trainer,
                                out_folder, m_lp)

            workers.append(agent_0)

        return workers

def get_1lvl_ac_rnn(env_gen, num_workers, out_folder):
    """Returns a group of hierarchical agent workers
    only 1 lvl in the hierarchy, a single ac"""

    with tf.device("/cpu:0"):
        lp = {'lambda'            : 1,
              'gamma'             : .99,
              'update_ival'       : np.inf,
              'max_episode_length': 400}

        trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99

        workers = []
        for i in range(num_workers):
            workers.append(AC_rnn_ra_Worker(env_gen(), 'agent_' + str(i),
                                            trainer, out_folder, lp))
        return workers

    
def get_1lvl_ac(env_gen, num_workers, out_folder):
    """Returns a group of hierarchical agent workers
    only 1 lvl in the hierarchy, a single ac"""
    
    with tf.device("/cpu:0"):
        lp = {'lambda'            : 1,
              'gamma'             : .99,
              'update_ival'       : np.inf,
              'max_episode_length': 10}

        trainer = tf.train.AdamOptimizer(learning_rate=0.00025) # beta1=0.99

        workers = []
        for i in range(num_workers):
            workers.append(AC_Worker(env_gen(), 'agent_' + str(i),
                                     trainer, out_folder, lp))

        return workers


