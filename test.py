##Simple function for running random things when debugging

from environments import env_factory
import matplotlib.pyplot as plt
import random

env = env_factory.get('VC_Waypoints')()
env.reset()
env.render()

for _ in range(10000):
    max_a = 0.1
    action = [random.uniform(-1*max_a, max_a), random.uniform(-1*max_a, max_a)]
    s, r, t = env.step(action)
    env.render()
    # plt.pause(.01)

    if t:
        plt.pause(1)
        env.reset()
        print('resetting state. r = ' + str(r))
    
        
