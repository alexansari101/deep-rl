import tensorflow as tf
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
            # m_env = AC_rnn_ra_Wrapper(env,i,s_shape, a_size, trainer,
            #                           global_episodes, max_episode_length,
            #                           update_ival, gamma, lam, args.output,
            #                           grid_size = grid_size)
            # workers.append(AC_Worker(m_env,i,m_s_shape,m_a_size,m_trainer,
            #                          args.output,global_episodes))
            # workers.append(AC_rnn_ra_Worker(env, 'agent_' + str(i),
            #                                 m_s_shape, a_size, trainer,
            #                                 args.output, global_episodes))

            agent_1 = AC_rnn_ra_Worker(env, 'agent_1_'+str(i), s_shape, a_size,
                                       trainer, out_folder, global_episodes,
                                       hlvl=1)
            env_1 = H_Env_Wrapper(agent_1, subgoal, global_episodes,
                                  max_ep_len=50, gamma=.9, lam=1,
                                  model_path=out_folder)
            # env_1.flags['verbose']=True
            
            agent_0 = AC_Worker(env_1, 'agent_0_'+str(i), m_s_shape, m_a_size, m_trainer,
                                out_folder, global_episodes)

            workers.append(agent_0)

        return workers

def get_1lvl_ac_rnn(env_gen, num_workers, out_folder):
    with tf.device("/cpu:0"):
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
                                            out_folder, global_episodes))

        return workers



if __name__ == '__main__':
    s_shape = [84,84,4]       # Observations are rgb frames 84 x 84 + goal
    a_size = 2                # planar real-valued accelerations
    m_s_shape = [84,84,3]
    m_a_size = 16      # Should be a square number
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                  trainable=False)


    AC_rnn_ra_Network(s_shape,a_size,'global_1',None)
    AC_Network(m_s_shape,m_a_size,'global_0',None)
    env = environments.waypoint_planner.gameEnv()
    agent = get_2lvl_HA3C(env, 1, 'tmp_folder', global_episodes)
    

