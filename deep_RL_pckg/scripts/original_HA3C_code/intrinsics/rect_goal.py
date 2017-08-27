from functools import reduce
from operator import mul
import numpy as np
import matplotlib.pyplot as plt
from .augmented_env_base import Augmented_Env_Base

class RectGoal(Augmented_Env_Base):
    """Defines subgoals and intrinsic rewards for hierarchy"""

    def __init__(self, env):
        Augmented_Env_Base.__init__(self, env)


        #To mimic the openAI gym enviornment
        self.m_action_space.n = 4


        sh = list(env.observation_space.shape)
        sh[2]+=1
        self.observation_space.shape = tuple(sh)

        self.mask = None

    def set_meta_action(self, m_a):
        self.mask = self.get_mask(m_a)

    def augment_obs(self, s):
        return self.get_meta_state(s, self.mask)

    # ARA - todo: add to critic and generalize to get "filtered actions"
    def get_mask(self,a,brdr=4):
        """Generate a rectangular mask.
        a = (x_left, y_lower, width, height)"""
        size = self.observation_space.shape[0:2]
        (x_max, y_max) = size

        x1 = int(np.clip(a[0]*x_max, 0, x_max-1))
        x2 = int(np.clip(x1+a[2]*x_max, x1, x_max-1))
        
        y1 = int(np.clip(a[1]*y_max, 0, y_max-1))
        y2 = int(np.clip(y1+a[3]*y_max, y1, y_max-1))
        
        mask = np.zeros(size)
        mask[x1:x2, y1:y2] = 1.0 # fill grid cell

        return mask

    def get_meta_state(self,s,g):
        """compute the 4 channel meta-state from meta-action (goal / option)
        Parameters
        ==========
        s: the raw state
        g: the goal mask
        """
        return np.dstack([s,g]) # stack state and goal

    def intrinsic_reward(self,s,a,sp,f,m_d):
        """Intrinsic reward from internal critic in hierarchical RL.

        Arguments:
        s: state/255 prior to taking action
        a: action
        sp: state/255 after taking action
        f: extrinsic reward
        g: current goal

        Returns:
        intrinsic_reward: reward based on agreement with the meta goal
        meta_reward: reward for the meta agent
        done: Terminal wrapped_env?
       """
        g = self.mask

        m_r = f

        g = self.mask
        m_r = f
        i_r = 0

        hero = s[:,:,2]
        herop = sp[:,:,2]

        done = m_d #Maybe want to make this more complex in the future


        if done and m_r < 0:
            return m_r, m_r, True

        # m_r = np.sum(g)/(76**2)

        if np.sum(g.astype(bool)*hero) > 3.5:
            i_r += m_r #Only give sub agent the m_r if it agrees with goal
            i_r += 0.05 #Make the sub agent go to the meta goal, even if there is no external reward
            m_r += 1

            # done = True


        # i_r = np.clip(i_r,-1,1)

        return i_r, m_r, done


    # Returns the current environment state
    def render_meta_state(self, g):
        if self.im is None:
            self.im = plt.imshow(self.get_last_obs())
            plt.ion()

        image = self.get_meta_state_image(g)
        self.im.set_data(image)
        plt.pause(0.0001)
        plt.draw()
        return image

    def get_meta_state_image(self, g):
        s = self.get_last_obs()
        m_s = self.get_meta_state(s,self.get_mask(g))
        return self.visualize_meta_state(m_s)

    def visualize(self,s):
        """
        return a 3-channel state frame for policy visulaization
        Parameters
        ==========
        s: a 4 channel state where the last channel is the meta_goal
        """
        sf = s[:,:,:-1].copy()
        sf[:,:,1] += 0.5*s[:,:,-1]
        sf[:,:,1] = np.clip(sf[:,:,1],0,1)
        return sf
