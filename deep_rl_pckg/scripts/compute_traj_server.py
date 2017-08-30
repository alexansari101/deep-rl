#!/usr/bin/env python

from deep_rl_pckg.srv import *
import rospy
from std_msgs.msg import String, Float32MultiArray

class traj_services:
	def __init__(self):
		self.x=4

	def callback(self, req):
	    print (self.x)
	    print (req.aqFunction)
	    pose_msg=Float32MultiArray(data=[1,2,3])
	    return computeTrajResponse(pose_msg)

	def add_two_ints_server(self):
	    rospy.init_node('compute_traj_server')
	    ser = rospy.Service('compute_traj', computeTraj, self.callback)
	    print "Ready to compute_traj..."
	    rospy.spin()

if __name__ == "__main__":
    serv = traj_services()
    serv.add_two_ints_server()
