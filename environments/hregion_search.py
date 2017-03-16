import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from .cregion import cRegion
        
class gameEnv():
    """Environment definition for hierarchical RL"""
    
    def __init__(self,v_max=1.0,a_max=1.0):        
        self.a_max = a_max
        self.v_max = v_max        
        self.num_goals = 1
        self.num_obstacles = 0
        self.hero = np.zeros(4)
        self.hero_old = self.hero.copy()
        self.goals = []
        self.brdr = 4
        self.width = 2
        self.state = np.zeros([84,84,3])
        plt.imshow(self.state,interpolation="nearest")

    def reset(self):
        self.state.fill(0)
        # add goals to background
        self.goals = []
        for i in range(self.num_goals):
            # WARNING: assumes high > low (may not be true)
            # w = np.random.randint(20,(84-2*self.brdr)//self.num_goals)
            w = 80-2*self.brdr
            if w % 2 != 0:
                w -= 1
            goal = np.random.randint(self.brdr+w//2+2, 84-self.brdr-w//2+1,
                                     size=2)
            goal[0]=42
            self.goals.append(np.append(goal,np.zeros(2)))
            goal = np.round(goal).astype(int)
            b = self.state[goal[0]-w//2:goal[0]+w//2,goal[1]-w//2:goal[1]+w//2,:]
            b.fill(0)
            reg = cRegion()
            b[:,:,1] = reg.image(size=[w,w],blur=2.5)

        # reset hero location
        self.hero = np.random.randint(self.brdr+self.width+2,
                                      83-self.brdr-self.width,
                                      size=2).astype(float)
        self.hero = np.append(self.hero,np.zeros(2))
            
        # add boarder
        brdr, b = self.brdr, self.state
        b[:brdr,:,0] = b[-brdr:,:,0] = b[:,:brdr,0] = b[:,-brdr:,0] = 255
        
        return self.renderEnv()

    def moveChar(self,accel):        
        self.hero_old = self.hero.copy()
        penalize = 0.0
        a_m = 10*self.a_max
        v_m = 10*self.v_max
        accel = a_m * np.tanh(np.asarray(accel)/self.a_max)
        self.hero[0] += self.hero[2]
        self.hero[1] += self.hero[3]
        vx = accel[-1] + .9*self.hero[3]
        vy = accel[-2] + .9*self.hero[2]
        self.hero[3] = v_m * np.tanh(vx/v_m)
        self.hero[2] = v_m * np.tanh(vy/v_m)
        return penalize

    def checkGoal(self):
        hy,hx = np.round(self.hero[:2]).astype(int)
        r = 0 # -0.05
        d = False
        width = self.width
        for goal in self.goals:
            gy,gx = np.round(goal[:2]).astype(int)
            if hx+width > 82-self.brdr or hx-width < 1+self.brdr:
                r = -10.0
                d = True
            elif  hy+width > 82-self.brdr or hy-width < 1+self.brdr:
                r = -10.0
                d = True
            else:
                nrm = 255.0*(2*width)**2
                a = self.state
                r += 2*np.sum(a[hy-width:hy+width, hx-width:hx+width,1])/nrm
                r *= np.exp(-np.linalg.norm(self.hero[2:]))
                r -= 2*np.sum(a[hy-width:hy+width, hx-width:hx+width,0])/nrm                
                d = False
        return r,d

    def renderEnv(self):
        width = self.width
        # render hero
        hero = np.round(self.hero).astype(int)
        hero_p = np.round(self.hero_old).astype(int)
        self.state[hero_p[0]-width:hero_p[0]+width,
                   hero_p[1]-width:hero_p[1]+width,2] = 0
        if True:
            self.state[hero[0]-width:hero[0]+width,
                       hero[1]-width:hero[1]+width,0] = 0
            hs = np.array(self.state[hero[0]-width:hero[0]+width,
                                     hero[1]-width:hero[1]+width,1]).astype(float)
            hs *= (1-np.exp(-np.linalg.norm(hero[2:])))
            self.state[hero[0]-width:hero[0]+width,
                       hero[1]-width:hero[1]+width,1] = hs.astype(int)
            
        self.state[hero[0]-width:hero[0]+width,
                   hero[1]-width:hero[1]+width,2] = 255        
        self.state = np.array(scipy.misc.toimage(self.state))
        return self.state

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done        


