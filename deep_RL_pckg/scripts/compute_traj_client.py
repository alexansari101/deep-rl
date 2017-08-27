#!/usr/bin/env python

import sys
import rospy
from beginner_tutorials.srv import *
from std_msgs.msg import String, Float32MultiArray


def add_two_ints_client(x):
	rospy.wait_for_service('compute_traj')
	print('Connected!!')
	try:
		compute_traj = rospy.ServiceProxy('compute_traj', computeTraj)
		resp1 = compute_traj(x)
		return resp1.poseAgent
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

def usage():
	return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
	print(add_two_ints_client(Float32MultiArray(data=[1,1,1,1])))