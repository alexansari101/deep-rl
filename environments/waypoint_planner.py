#!/usr/bin/env python3

""" Game Enviornment for a waypoint planner
This environment has multiple waypoints
Rewards are received after all waypoints have been traversed

TODO currenlty only one waypoint
"""

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
from PIL import Image

def genNumber(num):
    """generates a number as a numpy array image
    TODO!!!!!!!!!!!!
    Not implemented!!! Currently just returns a solid block. """
    path = os.path.join(os.path.dirname(__file__))
    path = path + '/' + str(num) + '.png'
    im = Image.open(path).convert('L').resize((12,12), Image.ANTIALIAS)
    # return 255*np.ones(4)
    return (np.asarray(im) < 200)*255

    
        
class gameEnv():
    """Environment definition for hierarchical RL"""
    
    def __init__(self,v_max=1.0,a_max=1.0):        
        self.a_max = a_max
        self.v_max = v_max        
        self.num_goals = 9
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

    def reset(self):
        self.state.fill(0)
        
        # add goals to background
        self.goals = []
        self.next_goal = 0
        
        for i in range(self.num_goals):
            # WARNING: assumes high > low (may not be true)
            # w = np.random.randint(20,(84-2*self.brdr)//self.num_goals)
            w = 4
            if w % 2 != 0:
                w -= 1
            num_width = 12
            gc = np.random.randint(self.brdr, 84-self.brdr-num_width,
                                     size=2)
            
            goal = np.zeros((84,84))
            # print(genNumber(i+1))
            # print(genNumber(i+1).shape)
            # print(goal[gc[0]:gc[0]+24, gc[1]:gc[1]+24].shape)
            goal[gc[0]:gc[0]+num_width, gc[1]:gc[1]+num_width] = genNumber(i+1)

            self.goals.append(goal)

            # b = self.state[goal[0]-w//2:goal[0]+w//2,goal[1]-w//2:goal[1]+w//2,:]
            # b.fill(0)
            # b[:,:,1] = genNumber(i+1)


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
        self.hero_old = self.hero.copy()
        penalize = 0.0
        a_m = 10*self.a_max
        v_m = 10*self.v_max
        accel = a_m * np.tanh(np.asarray(accel_in)/self.a_max)
        self.hero[0] += self.hero[2]
        self.hero[1] += self.hero[3]
        vx = accel[-1] + .9*self.hero[3]
        vy = accel[-2] + .9*self.hero[2]
        self.hero[3] = v_m * np.tanh(vx/v_m)
        self.hero[2] = v_m * np.tanh(vy/v_m)

        if np.isnan(self.hero[3]):
            print('hero[3] is nan')
            print(a_m)
            print(v_m)
            print(accel)
            print(accel_in)
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
        # plt.draw()
        plt.pause(0.0001)
        plt.draw()
        return image

    def getState(self):
        width = self.width
        state = self.state.copy()
        # render hero
        hero = np.round(self.hero).astype(int)
        hero_p = np.round(self.hero_old).astype(int)
        # state[hero_p[0]-width:hero_p[0]+width,
        #       hero_p[1]-width:hero_p[1]+width,2] = 0
        state[hero[0]-width:hero[0]+width,
              hero[1]-width:hero[1]+width,0] = 0
            # hs = np.array(self.state[hero[0]-width:hero[0]+width,
            #                          hero[1]-width:hero[1]+width,1]).astype(float)
            # hs *= (1-np.exp(-np.linalg.norm(hero[2:])))
            # self.state[hero[0]-width:hero[0]+width,
            #            hero[1]-width:hero[1]+width,1] = hs.astype(int)
            
        state[hero[0]-width:hero[0]+width,
              hero[1]-width:hero[1]+width,2] = 255


        for i in range(self.next_goal, self.num_goals):
            goal = self.goals[i]
            state[:,:,1] += goal
        state = np.clip(state,0,255)
        state = np.array(scipy.misc.toimage(state))
        return state

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.getState()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done        


