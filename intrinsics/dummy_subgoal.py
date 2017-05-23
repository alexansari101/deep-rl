from functools import reduce
from operator import mul
import numpy as np
import matplotlib.pyplot as plt
from util import process_frame
from .augmented_env_base import Augmented_Env_Base

class DummyGoal(Augmented_Env_Base):
    """Defines subgoals and intrinsic rewards for hierarchy
    This is a dummy subgoal which does not alter the environment
    The sub agent recieves the reward directly from the environment"""

    def __init__(self, env):
        Augmented_Env_Base.__init__(self, env)

        #To mimic the openAI gym enviornment
        self.m_action_space.n = 16 #product of grid_size
        self.observation_space = env.observation_space


    def set_meta_action(self, m_a):
        pass

    def augment_obs(self, s):
        return s

    def intrinsic_reward(self,s,a,sp,f,m_d):
        """Intrinsic reward from internal critic in hierarchical RL.
       """
        return f, f, m_d


    # Returns the current environment state
    def render_meta_state(self, g):
        if self.im is None:
            self.im = plt.imshow(self.get_last_obs())
            plt.ion()

        image = self.get_last_obs()
        self.im.set_data(image)
        plt.pause(0.0001)
        plt.draw()
        return image

    def visualize(self, s):
        return s
