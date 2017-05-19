from functools import reduce
from operator import mul
import numpy as np
import matplotlib.pyplot as plt
from util import process_frame

class GridGoal():
    """Defines subgoals and intrinsic rewards for hierarchy"""

    def __init__(self, raw_ob_shape, grid_size):

        self.grid_size = grid_size

        #To mimic the openAI gym enviornment
        self.action_space = lambda: None
        self.action_space.n = reduce(mul,grid_size) #product of grid_size

        self.observation_space = lambda: None
        sh = list(raw_ob_shape)
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
        g: current goal
        Returns:
        intrinsic_reward: reward based on agreement with the meta goal
        meta_reward: reward for the meta agent
        done: Terminal wrapped_env?
       """
        g = self.mask
        m_r = f

        done = False
        r = -0.05
        # r = 0.0  

        
        if m_d:
            done = True
            if f<0:
                r = f

        #small reward for moving slowly
        if np.sum(s[:,:,2].astype(bool)*sp[:,:,2]) > 0:
            r += 0.05

        #large reward if the agent's past and present
        #  state is inside the masked region

        if np.sum(g.astype(bool)*s[:,:,2]) > 3.5 \
             and np.sum(g.astype(bool)*sp[:,:,2]) > 3.5:
            r += 1

            done = True


        i_r = np.clip(r,-1,1)

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
        sf[:,:,1] /= np.max(sf[:,:,1])
        return sf
