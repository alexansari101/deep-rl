#!/usr/bin/env python3

""" Game Enviornment for a waypoint planner
This environment has multiple waypoints
Rewards are received after all waypoints have been traversed

"""

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
from PIL import Image

def genNumber(num, width):
	"""Generates a number as a picture in a numpy array
	"""
	path = os.path.join(os.path.dirname(__file__))
	path = path + '/' + str(num) + '.png'
	im = Image.open(path).convert('L').resize((width,width), Image.ANTIALIAS)
	return (np.asarray(im) < 200)*255

	
		
class gameEnv():
	"""Environment definition for hierarchical RL"""
	
	def __init__(self,v_max=1.0,a_max=1.0, num_goals=1):
		self.a_max = a_max
		self.v_max = v_max        
		self.num_goals = num_goals
		self.next_goal = 0
		self.num_obstacles = 0
		self.hero = np.zeros(4)
		self.hero_old = self.hero.copy()
		self.goals = []
		self.brdr = 4
		self.width = 2
		self.state = np.zeros([84,84,3])
		self.im = None
		plt.imshow(self.state,interpolation="nearest")

		#To mimic the openAI gym environment
		self.action_space = lambda: None
		self.action_space.n = 2
		self.observation_space = lambda: None
		self.observation_space.shape = (84,84,3)

	def reset(self):
		self.state.fill(0)
		
		# add goals to background
		self.goals = []
		self.next_goal = 0
		
		for i in range(self.num_goals):
			w = 4
			if w % 2 != 0:
				w -= 1
			goal_width = 24
			gc = np.random.randint(self.brdr, 84-self.brdr-goal_width,
									 size=2)
			goal = np.zeros((84,84))
			goal[gc[0]:gc[0]+goal_width, gc[1]:gc[1]+goal_width] = genNumber(i+1, goal_width)
			self.goals.append(goal)


		# reset hero location
		self.hero = np.random.randint(self.brdr+self.width+2,
									  83-self.brdr-self.width,
									  size=2).astype(float)
		self.hero = np.append(self.hero,np.zeros(2))
			
		# add boarder
		brdr, b = self.brdr, self.state
		b[:brdr,:,0] = b[-brdr:,:,0] = b[:,:brdr,0] = b[:,-brdr:,0] = 255
		
		return self.getState()

	def moveChar(self,accel_in):
		"""Applies the acceleration command to the character state"""

		self.hero_old = self.hero.copy()
		penalize = 0.0
		a_m = 10*self.a_max
		v_m = 1*self.v_max
		accel = a_m * np.tanh(np.asarray(accel_in)/self.a_max)
		self.hero[0] += self.hero[2]
		self.hero[1] += self.hero[3]
		vx = accel[-1] + .9*self.hero[3]
		vy = accel[-2] + .9*self.hero[2]
		self.hero[3] = v_m * np.tanh(vx/v_m)
		self.hero[2] = v_m * np.tanh(vy/v_m)

		############## position control 
		# if action==0: #right
		# 	self.hero[1] += 1
		# elif action==1: #left 
		# 	self.hero[1] -= 1
		# elif action==2: #up (moving in image space --> origin at upper left corner)
		# 	self.hero[0] -= 1
		# elif action==3: #down 
		# 	self.hero[0] += 1               
		
		return penalize

	def borderCollision(self):
		"""Returns true if the hero has collided with the border"""
		width = self.width
		hy,hx = np.round(self.hero[:2]).astype(int)

		# if hx+width > 100:
		#     #Trying to debug overflow error
		#     print(hx+width)
		np.seterr(all='raise')
		try:
			if hx+width > 82-self.brdr or hx-width < 1+self.brdr:
				np.seterr(all='print')
				return True
			if  hy+width > 82-self.brdr or hy-width < 1+self.brdr:
				np.seterr(all='print')
				return True
		except:
			print(self.hero)
			print(self.hero_old)
			print(hx)
			print(hy)
			print(width)
			print(self.brdr)
			raise
		np.seterr(all='print')
		return False
	
	def checkGoal(self):
		"""Computes the reward the hero receives
		-negative an terminal if crash
		-positive if final goal reached
		
		Returns
		=======
		r,d
		r: numerical reward
		d: boolean - True if a terminal state
		"""
		hy,hx = np.round(self.hero[:2]).astype(int)
		r = 0 # -0.05
		d = False
		width = self.width

		if self.borderCollision():
			return -1, True

		goal = self.goals[self.next_goal]
		#NEEDS EDIT
		# gy,gx = np.round(goal[:2]).astype(int)
		reached = np.sum(goal[hy-width:hy+width, hx-width:hx+width]) > 0.5
		if reached:
			self.next_goal += 1
		if self.next_goal == len(self.goals):
			r = 1
			d = True

		return r,d

	def render(self):
		if self.im is None:
			self.im = plt.imshow(self.getState())
			plt.ion()

		image = self.getState()
		self.im.set_data(image)
		plt.pause(0.0001)
		plt.draw()
		return image

	def getState(self):
		"""Returns the last observation"""
		width = self.width
		state = self.state.copy()

		hero = np.round(self.hero).astype(int)
		hero_p = np.round(self.hero_old).astype(int)

		#Add hero to the image on top of everything else
		state[hero[0]-width:hero[0]+width,
			  hero[1]-width:hero[1]+width,0] = 0
		state[hero[0]-width:hero[0]+width,
			  hero[1]-width:hero[1]+width,2] = 255

		#Overlay the goal regions
		# for i in range(self.next_goal, self.num_goals):
		#     goal = self.goals[i]
		#     state[:,:,1] += goal
		#Overlay only the next goal
		if(self.next_goal < len(self.goals)):
			state[:,:,1] += self.goals[self.next_goal]

		state = np.clip(state,0,255)
		state = np.array(scipy.misc.toimage(state))
		return state

	def step(self,action):
		penalty = self.moveChar(action)
		reward,done = self.checkGoal()
		state = self.getState()
		return state,(reward+penalty),done        


