import tensorflow as tf
from agents.ac_network import AC_Network
from agents.ac_rnn_ra_network import AC_rnn_ra_Network
from agents.ac_worker import AC_Worker
from agents.ac_rnn_ra_worker import AC_rnn_ra_Worker
import environments
from environments.h_env_wrapper import H_Env_Wrapper

def get_2lvl_HA3C(env, i, out_folder, global_episodes):

    """Returns a hierarchical agent
    lvl 1 is AC
    lvl 2 is AC_RNN_RA"""
    trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
    s_shape = [84,84,4]       # Observations are rgb frames 84 x 84 + goal
    a_size = 2                # planar real-valued accelerations
    m_s_shape = [84,84,3]
    m_a_size = 16      # Should be a square number


    agent_1 = AC_rnn_ra_Worker(env, 'agent_1_'+str(i), s_shape, a_size,
                               trainer, out_folder, global_episodes,
                               hlvl=1)
    env_1 = H_Env_Wrapper(agent_1, global_episodes,
                          max_ep_len=50, gamma=.9, lam=1,
                          model_path=out_folder,
                          grid_size=(4,4))
    # env_1.flags['verbose']=True

    agent_0 = AC_Worker(env_1, 'agent_0_'+str(i), m_s_shape, m_a_size, trainer,
                        out_folder, global_episodes)
    return agent_0


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
    

