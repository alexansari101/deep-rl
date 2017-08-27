##Simple function for running random things when debugging

from environments import env_factory
import matplotlib.pyplot as plt
import random

env = env_factory.get('Waypoints')()
env.reset()
env.render()

# for _ in range(10000):
#     max_a = 0.1
#     action = [random.uniform(-1*max_a, max_a), random.uniform(-1*max_a, max_a)]
#     s, r, t = env.step(action)
#     env.render()
#     # plt.pause(.01)

#     if t:
#         plt.pause(1)
#         env.reset()
#         print('resetting state. r = ' + str(r))
    
        
key_to_action = {'d':[0,1],
				 'a':[0,-1],
				 'w':[-1,0],
				 's':[1,0]}

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