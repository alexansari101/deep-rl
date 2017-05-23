#!/usr/bin/env python3

""" 
Stupidly simple game which gives reward if action 7 is chosen.
This is to ensure ac can learn

"""

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
from PIL import Image

        
class gameEnv():
    """Environment definition for hierarchical RL"""
    
    def __init__(self,v_max=10.0,a_max=1.0):

        self.brdr = 4
        self.width = 2
        self.state = np.zeros([84,84,3])
        self.im = None

        #To mimic the openAI gym environment
        self.action_space = lambda: None
        self.action_space.n = 16
        self.observation_space = lambda: None
        self.observation_space.shape = (84,84,3)

    def reset(self):
        self.state.fill(0)
        brdr, b = self.brdr, self.state
        b[:brdr,:,0] = b[-brdr:,:,0] = b[:,:brdr,0] = b[:,-brdr:,0] = 255

        return self.getState()


    def render(self):
        return self.state

    def getState(self):
        return np.array(scipy.misc.toimage(self.state))

    def step(self,action):
        state = self.getState()
        return self.getState(), int(action==7), False


