##Simple function for running random things when debugging

import environments.waypoint_planner as envs
from environments.ac_rnn_ra_wrapper import AC_rnn_ra_Wrapper
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from agents.ac_rnn_ra_network import AC_rnn_ra_Network


m_s_shape = [84,84,3]
s_shape = [84,84,4]       # Observations are rgb frames 84 x 84 + goal
a_size = 2
trainer = tf.train.AdamOptimizer(learning_rate=1e-5) # beta1=0.99
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                              trainable=False)
m_max_episode_length = 10
max_episode_length = 50
update_ival = np.inf      # train after this many steps if < max_episode_length
gamma = .99               # discount rate for reward discounting
lam = 1                   # .97; discount rate for advantage estimation
model_path = './model'

tf.reset_default_graph()
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                              trainable=False)
master_network = AC_rnn_ra_Network(s_shape,a_size,'global_1',None)

    

env = envs.gameEnv()
m_env = AC_rnn_ra_Wrapper(env,0,s_shape, a_size, trainer,
                          global_episodes, max_episode_length,
                          update_ival, gamma, lam, model_path)
m_env.reset()
# m_env.render()
m_env.flags['render'] = False
m_env.flags['verbose'] = False
m_env.flags['train'] = True
# plt.pause(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        max_a = 0.1
        # action = [random.uniform(-1*max_a, max_a), random.uniform(-1*max_a, max_a)]
        action = random.randint(0,15)
        s, r, t = m_env.step(action)
        # m_env.render_meta_state(action)
        # plt.pause(.01)
        sess.run(global_episodes.assign_add(1))
        
        if t:
            # plt.pause(1)
            env.reset()
            # print('resetting state. r = ' + str(r))
    
        
