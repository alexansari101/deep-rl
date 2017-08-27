#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simulates a saved A3C agent with an rnn and meta-learning.

The agent's rnn is fed with additional rewards and actions for meta-learning.

Todo:
    * Consider encapsulating code into a base clase which can be derived from
      for other, specialized simulation needs.
    * Add input arguments and a main function.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from environments.hregion_search import gameEnv
from agents.ac_rnn_ra_network import AC_rnn_ra_Network
from util import process_frame

###################
# Hyperparameters #
###################
max_episode_length = 50
s_shape = [84,84,3]     # Observations are rgb frames 84 x 84
a_size = 2              # Agent can move in a line
gameArgs = {}
load_model = False
model_path = './model'


#######################
# Simulate the Policy #
#######################
env_g = gameEnv(**gameArgs);
s = env_g.reset()
sarray = [s]
rarray = []

tf.reset_default_graph()

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                  trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5)
    ac_net = AC_rnn_ra_Network(s_shape,a_size,'global_0',None)
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    rnn_state = ac_net.state_init
    i=0
    d=False
    r = 0
    a = np.array([0,0])
    while i < max_episode_length and d == False:
        s_p = process_frame(s)
        # Take an action using probabilities from policy network output.
        a,v,rnn_state = sess.run([ac_net.sample_a,
                                  ac_net.value,
                                  ac_net.state_out], 
                                 feed_dict={ac_net.inputs:[s_p],
                                            ac_net.prev_actions:[a],
                                            ac_net.prev_rewards:[[r]],
                                            ac_net.is_training_ph:False,
                                            ac_net.state_in[0]:rnn_state[0],
                                            ac_net.state_in[1]:rnn_state[1]})
        
        s,r,d = env_g.step(a)
        sarray.append(s)
        rarray.append(r)
        i += 1

print(sum(rarray))


################################
# Animate and Show the Results #
################################
im = plt.imshow(sarray[0],animated=True);

def updatefig(i):
    im.set_data(sarray[i]);
    return im,

anim = animation.FuncAnimation(plt.gcf(), updatefig, frames=len(sarray),
                               interval=75, blit=True)
# anim.save('./frames/a3c_'+str(sum(rarray))+'.mp4')
plt.show()
# HTML(anim.to_html5_video())
# plt.close(fig)
