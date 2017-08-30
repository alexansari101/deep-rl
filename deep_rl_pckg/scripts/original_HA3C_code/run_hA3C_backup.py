#!/usr/bin/env python

import matplotlib.pyplot as plt
from util import dir_utils
# from agents.ac_worker import AC_Worker
import environments
from environments.ac_rnn_ra_wrapper import AC_rnn_ra_Wrapper
from environments import env_factory
import argparse
import numpy as np
import tensorflow as tf
import threading
import multiprocessing
from agents.ac_network import AC_Network
from agents.ac_rnn_ra_network import AC_rnn_ra_Network
from agents.ac_rnn_ra_worker import AC_rnn_ra_Worker
from agents import H_Workers
from IPython import embed
from IPython import embed
from deep_rl_pckg.srv import*
import rospy
from std_msgs.msg import String, Float32MultiArray

class main_callback:
	def __init__(self):
		self.sess=None
		self.workers=None
	def process_args(self, args):
		""" Additional proccessing for args to load the correct folders for storage"""
		# if(args.load):
		#     args.output, iter_num = dir_utils.get_saved(args.output, args.env, args.trial, args.iter)
		#     args.iter = iter_num
		# else:
		
		args.output = dir_utils.get_output_folder(args.output, args.env, args.load, args.trial,
												  tmp=args.tmp)
			
		return args


	
	def callback_work(self,req):
		print(req.aqFunction)
		# embed()
		self.workers[0].test(self.sess,req.aqFunction,0)
		pose_msg=Float32MultiArray(data=[1,2,3])
		return computeTrajResponse(pose_msg)



	def main(self):  # noqa: D103
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
		parser.add_argument('--tmp', action='store_const', const=True)
		parser.add_argument('--trial', default=None, type=int, help='The trial number to load')
		parser.add_argument('--iter', default=0, type=int, help='The iter to load CURRENTLY UNUSED')
		parser.add_argument('--grid', default=4, type=int, help='Number of grid squares in a row or column')

		args = parser.parse_args()
		args = self.process_args(args)

		if args.train and not args.load:
			dir_utils.copy_files(args.output)

		if not args.tmp and args.train and not args.load:
			dir_utils.write_readme(args.output)

		if args.test:
			args.env = args.env + "_exp"
			
		print(args.env)
		# num_workers = multiprocessing.cpu_count() # number of available CPU threads
		num_workers = 8 #Hardcode num-workers for consistency across machines
		workers = H_Workers.get_2lvl_HA3C(env_factory.get(args.env), num_workers, args.output)
		self.workers = workers
		# workers = H_Workers.get_2lvl_rnn_ra_x2(env_factory.get(args.env), num_workers, args.output)
		# workers = H_Workers.get_dummy_2lvl_HA3C(env_factory.get(args.env), num_workers, args.output)
		# workers = H_Workers.get_1lvl_ac_rnn(env_factory.get(args.env), num_workers, args.output)
		# workers = H_Workers.get_1lvl_ac(env_factory.get(args.env), num_workers, args.output)

		saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

		with tf.Session() as self.sess:

			if(args.load):
				# print('Loading Model ' + args.output)
				ckpt = tf.train.get_checkpoint_state(args.output)
				print('Loading Model ' + ckpt.model_checkpoint_path)
				saver.restore(self.sess,ckpt.model_checkpoint_path)
			else:
				self.sess.run(tf.global_variables_initializer())

			if(args.train):

				worker_threads = []
				coord = tf.train.Coordinator()        
				for worker in workers:
					worker_work = lambda: worker.work(self.sess,coord,saver)
					t = threading.Thread(target=(worker_work))
					t.start()
					worker_threads.append(t)
				coord.join(worker_threads)

			if(args.debug):
				#Random things for debugging help
				worker_threads = []
				coord = tf.train.Coordinator()
				for worker in workers:
					worker.env.sess = self.sess
					worker_work = lambda: loop_stepping(worker, coord)
					t = threading.Thread(target=(worker_work))
					t.start()
					worker_threads.append(t)
				coord.join(worker_threads)
				
				
			if(args.test):
				rospy.init_node('compute_traj_server')
				s = rospy.Service('compute_traj', computeTraj, self.callback_work)
				print ('Ready to compute_traj...')
				rospy.spin()
				# for i in range(1):
				#   workers[0].test(sess,i)

			if(args.play):
				key_to_action = {'d':[0,1],
								 'a':[0,-1],
								 'w':[-1,0],
								 's':[1,0]}

				env = env_factory.get(args.env)()
				env.reset()
				env.render()
				episode_r = 0
				while True:
					print('Input a key:  \n')
					line = input('')
					s, r, d = env.step(key_to_action[line])
					env.render()
					episode_r += r
					if d:
						plt.pause(1)
						print('final state. Episode reward: ' + str(episode_r))
						episode_r = 0
						env.reset()
						env.render()

			
					

	def loop_stepping(self, worker, coord):
		while not coord.should_stop():
			worker.env.reset()
			worker.env.step(0)
			
if __name__ == '__main__':
	# rospy.init_node('RL_agent', anonymous=True)
	serv = main_callback()
	serv.main()

			
			

	


	
