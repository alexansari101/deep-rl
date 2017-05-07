from util import dir_utils
from agents.ac_worker import AC_Worker
import environments
from environments.ac_rnn_ra_wrapper import AC_rnn_ra_Wrapper
import argparse
import numpy as np
import tensorflow as tf
import threading
import multiprocessing
from agents.ac_network import AC_Network
from agents.ac_rnn_ra_network import AC_rnn_ra_Network
from agents.ac_rnn_ra_worker import AC_rnn_ra_Worker
import HA3C_2lvl


def process_args(args):
    """ Additional proccessing for args to load the correct folders for storage"""
    # if(args.load):
    #     args.output, iter_num = dir_utils.get_saved(args.output, args.env, args.trial, args.iter)
    #     args.iter = iter_num
    # else:
    
    args.output = dir_utils.get_output_folder(args.output, args.env, args.load, args.trial)
        
    return args

def load_env(env_name, ):
    if env_name == "Waypoints":
        return environments.waypoint_planner.gameEnv()
    if env_name == "Search":
        return environments.hregion_search.gameEnv()
    if env_name == "Center":
        return environments.center_goal.gameEnv()
    if env_name == "RandRegion":
        return environments.random_goal.gameEnv()
    
    raise ValueError('Unknown environment name: ' + str(env_name))



def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Runs hA3C')
    parser.add_argument('--env', default='Waypoints', help='env name: either Search or Waypoints')
    parser.add_argument(
        '-o', '--output', default='trials', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--train', action='store_const', const=True)
    parser.add_argument('--play', action='store_const', const=True, help='Manually control agent')
    parser.add_argument('--test', action='store_const', const=True)
    parser.add_argument('--debug', action='store_const', const=True)
    parser.add_argument('--load', action='store_const', const=True)
    parser.add_argument('--trial', default=None, type=int, help='The trial number to load')
    parser.add_argument('--iter', default=0, type=int, help='The iter to load CURRENTLY UNUSED')
    parser.add_argument('--grid', default=4, type=int, help='Number of grid squares in a row or column')


    args = parser.parse_args()
    args = process_args(args)


    max_episode_length = 50
    update_ival = np.inf      # train after this many steps if < max_episode_length
    gamma = .99               # discount rate for reward discounting
    lam = 1                   # .97; discount rate for advantage estimation
    s_shape = [84,84,4]       # Observations are rgb frames 84 x 84 + goal
    a_size = 2                # planar real-valued accelerations
    m_max_episode_length = 1
    m_s_shape = [84,84,3]
    m_a_size = args.grid**2      # Should be a square number

    

    # grid_size = (int(np.sqrt(m_a_size)), int(np.sqrt(m_a_size)))
    grid_size = (args.grid, args.grid)
    
    with tf.device("/cpu:0"):
        m_trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        trainer = tf.train.AdamOptimizer(learning_rate=0.00001) # beta1=0.99
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',
                                      trainable=False)
        
        # m_master_network = AC_Network(m_s_shape,m_a_size,'global_0',None) # meta network
        # master_network = AC_rnn_ra_Network(m_s_shape,a_size,'global_1',None)
        master_network = AC_rnn_ra_Network(m_s_shape,a_size,'global_0',None)

        num_workers = multiprocessing.cpu_count() # number of available CPU threads
        workers = []
        
        for i in range(num_workers):
            env = load_env(args.env)
            # m_env = AC_rnn_ra_Wrapper(env,i,s_shape, a_size, trainer,
            #                           global_episodes, max_episode_length,
            #                           update_ival, gamma, lam, args.output,
            #                           grid_size = grid_size)
            # workers.append(AC_Worker(m_env,i,m_s_shape,m_a_size,m_trainer,
            #                          args.output,global_episodes))
            workers.append(AC_rnn_ra_Worker(env, 'agent_' + str(i),
                                            m_s_shape, a_size, trainer,
                                            args.output, global_episodes))
            # meta_agent = HA3C_2lvl.get_2lvl_HA3C(env, i, args.output, global_episodes,
            #                                      trainer, m_trainer)
            # workers.append(meta_agent)
            
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:

        if args.load:
            # print('Loading Model ' + args.output)
            ckpt = tf.train.get_checkpoint_state(args.output)
            print('Loading Model ' + ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        if(args.train):
            worker_threads = []
            coord = tf.train.Coordinator()        
            for worker in workers:
                worker_work = lambda: worker.work(m_max_episode_length,update_ival,gamma,
                                                  lam,master_network,sess,coord,saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)
            
        if(args.test):
            for i in range(100):
                workers[0].evaluate(sess)


        if(args.play):
            key_to_action = {'d':[0,1],
                             'a':[0,-1],
                             'w':[-1,0],
                             's':[1,0]}

            env = load_env(args.env)
            env.reset()
            env.render()
            episode_r = 0
            while True:
                line = input('')
                s, r, d, = env.step(key_to_action[line])
                env.render()
                episode_r += r
                if d:
                    print('final state. Episode reward: ' + str(episode_r))
                    episode_r = 0
                    env.reset()
                    env.render()
                

            
            
if __name__ == '__main__':
    main()

            
            

    


    
