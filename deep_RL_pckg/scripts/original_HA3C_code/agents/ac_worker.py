# -*- coding: utf-8 -*-

"""Actor-critic workers

This module contains a worker agent that uses actor-critic networks.  By
threading the workers work() function they can be run asynchronously as
in A3C.  Each worker contains a copy of the AC network that it syncs
with a global network as in A3C.  The network specifies the policy the
worker uses to send actions to its copy of the environment.  The worker
stores the resulting rewards, observations, and any other variables
required for training the AC network.  The worker continuously loops
through through episodes and computes the graidents required to train
its local copy of the AC network, then updates the global network before
it re-syncs its local copy.  A worker can save summary statistics and
can visualize policy performance, e.g., by making animations, at desired
intervals.

Example:
    Using tensorboard to display stats during training:
    $ tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',\
      worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',\
      worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7'

Todo:
    * See AC_rnn_ra_Worker() todo list

"""

import numpy as np

import tensorflow as tf
import time
from datetime import timedelta

from util import update_target_graph, process_frame, discount
from agents.ac_network import AC_Network
from agents.ac_agent_base import AC_Agent_Base

class AC_Worker(AC_Agent_Base):
    """Simple advantage actor-critic worker with discrete actions.
    
    """
    
    def __init__(self,env,name,trainer,model_path,
                 learning_params, hlvl=0):
        """Initialize the worker environment, AC net, and trainer.

        Args:
            env: An environment object
            name (str): name of the worker agent.
            s_shape (list): shape of received environment states (observations)
            a_size (int): the dimension of the continuous action vector.
            trainer: a tensorflow optimizer from the tf.train module.
            model_path: folder under which to save the model
            learning_params: dictionary of parameters related to training
            hlvl: hierarchy level (0 for highest lvl agent)
        """
        AC_Agent_Base.__init__(self, env, name, trainer, model_path,
                               learning_params, hlvl)

        # Create the local copy of the network and the tensorflow op to
        # copy global paramters to local network
        s_shape = env.observation_space.shape
        a_size = env.action_space.n
        self.local_AC = AC_Network(s_shape,a_size,self.name,trainer, hlvl)
        self.update_local_ops = update_target_graph('global_'+str(hlvl),self.name)  


    def train(self,rollout,sess,gamma,lam,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3] # ARA - currently unused
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use
        # them to generate the advantage and discounted returns.  The
        # advantage function uses "Generalized Advantage Estimation"
        # Based on: https://github.com/awjuliani/DeepRL-Agents
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:]-self.value_plus[:-1]
        advantages = discount(advantages,gamma*lam)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     # ARA - using np.stack to support Ndarray states
                     self.local_AC.inputs:np.stack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
                                          self.local_AC.policy_loss,
                                          self.local_AC.entropy,
                                          self.local_AC.grad_norms,
                                          self.local_AC.var_norms,
                                          self.local_AC.apply_grads],
                                         feed_dict=feed_dict)
        return v_l/len(rollout),p_l/len(rollout),e_l/len(rollout),g_n,v_n

    def reset_agent(self):
        pass

    def start_trial(self):
        pass

    def sample_av(self, s, sess, prev_r):
        a_dist,v = sess.run([self.local_AC.policy,
                             self.local_AC.value], 
                            feed_dict={self.local_AC.inputs:[s]})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a,v
        
    def work(self,sess,coord,saver):
        gamma = self.gamma
        lam = self.lam

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        t0 = time.time()
        print("Starting worker " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                state_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)

                while d == False:
                    # Take an action using probabilities from policy
                    # network output.

                    a, v = self.sample_av(s, sess, None)

                    s1,r,d = self.env.step(a)


                    episode_frames.append(s1)
                    s1 = process_frame(s1)
                    if episode_step_count == self.max_ep - 1:
                        d = True
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    state_values.append(v[0,0])

                    episode_reward += r
                    s = s1

                    total_steps += 1
                    episode_step_count += 1
                                        
                    # If the episode hasn't ended, but the experience
                    # buffer is full, then we make an update step using
                    # that experience rollout.
                    if len(episode_buffer) == self.update_ival and \
                       d != True and \
                       episode_step_count != self.max_ep - 1:
                        # Since we don't know what the true final return
                        # is, we "bootstrap" from our current value
                        # estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(sess,gamma,lam,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)


                
                # Update the network using the experience buffer at the
                # end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,
                                                     sess,gamma,lam,0.0)

                if episode_count % 100 == 0 and self.is_writer:
                    saver.save(sess,self.model_path+'/model.ckpt', episode_count)
                    self.evaluate(sess, episode_count)
                    
                    s_dt = str(timedelta(seconds=time.time()-t0))
                    print("Saved Model " + str(episode_count) + '\tat time ' + s_dt)

                    
                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:

                
                    data = {'Perf/Reward'       : episode_reward,
                            'Perf/Length'       : episode_step_count,
                            'Perf/Value'        : np.mean(state_values),
                            'Perf/Global Ep'    : sess.run(self.global_episodes),
                            'Losses/Value Loss' : v_l,
                            'Losses/Policy Loss': p_l,
                            'Losses/Entropy'    : e_l,
                            'Losses/Grad Norm'  : g_n,
                            'Losses/Var Norm'   : v_n}
                    self.write_summary(data, episode_count)

                if self.is_writer:
                    sess.run(self.increment)
                episode_count += 1
