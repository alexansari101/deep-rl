from functools import reduce
from operator import mul
import numpy as np
import matplotlib.pyplot as plt
from util import process_frame
from .augmented_env_base import Augmented_Env_Base

class GridGoal(Augmented_Env_Base):
    """Defines subgoals and intrinsic rewards for hierarchy"""

    def __init__(self, env, grid_size):
        Augmented_Env_Base.__init__(self, env)

        self.grid_size = grid_size

        #To mimic the openAI gym enviornment
        self.m_action_space.n = reduce(mul,grid_size) #product of grid_size


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
        """Generate a mask with rectangular cell."""
        size = self.observation_space.shape[0:2]
        ni,nj = self.grid_size # divide a mask into a cell regions
        i = np.round(np.linspace(0,size[0]-2*brdr,ni+1)).astype(int)+brdr
        j = np.round(np.linspace(0,size[1]-2*brdr,nj+1)).astype(int)+brdr

        ri = a // ni # a should be between 0 and np.prod(grid)
        rj = a % ni
        mask = np.zeros(size)
        mask[i[ri]:i[ri+1],j[rj]:j[rj+1]] = 1.0 # fill grid cell

        if(np.max(mask)==0):
            print('a: ' + str(a) + '  ri: ' + str(ri) + '  rj: ' + str(rj))
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
        m_d: terminal state
        Returns:
        intrinsic_reward: reward based on agreement with the meta goal
        meta_reward: reward for the meta agent
        done: Terminal wrapped_env?
       """
        g = self.mask
        m_r = f
        i_r = 0

        hero = s[:,:,2]
        herop = sp[:,:,2]

        done = m_d #Maybe want to make this more complex in the future


        if done and m_r < 0:
            return m_r, m_r, True


        #small reward for moving slowly
        # if np.sum(hero.astype(bool)*herop) > 0:
        #     i_r += 0.05

        #large reward if the agent's past and present
        #  state is inside the masked region

        # if np.sum(g.astype(bool)*hero) > 3.5 \
        #      and np.sum(g.astype(bool)*herop) > 3.5:
        if np.sum(g.astype(bool)*hero) > 3.5:
            # r += 1
            # i_r += np.sum(hero.astype(bool)*herop)
            i_r += m_r #Only give sub agent the m_r if it agrees with goal
            i_r += 0.05 #Make the sub agent go to the meta goal, even if there is no external reward

            # done = True


        # i_r = np.clip(i_r,-1,1)

        return i_r, m_r, done

        #Same Reward as Alex Used
        # done = False
        # r = -0.05
        # # if the agent's past and present state is inside the masked region
        # if f < 0:
        #     done = True
        #     r = -10
        # elif np.sum(g.astype(bool)*s[:,:,2]) > 3.5 \
        #      and np.sum(g.astype(bool)*sp[:,:,2]) > 3.5:
        #     f_diff = np.sum(sp[:,:,2]-s[:,:,2])/4
        #     if(f_diff > 0.0001):
        #         print(f_diff)
        #     r += 0.35*np.exp(-f_diff)
        #     if r > 0:
        #         done = True
        # return r, done    




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
