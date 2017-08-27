#!/usr/bin/env python

from beginner_tutorials.srv import *
import rospy
from std_msgs.msg import String, Float32MultiArray


def deepRL(req):
    print (req.aqFunction)
    pose_msg=Float32MultiArray(data=[1,2,3])
    return computeTrajResponse(pose_msg)

def add_two_ints_server():
    rospy.init_node('compute_traj_server')
    s = rospy.Service('compute_traj', computeTraj, deepRL)
    print "Ready to compute_traj..."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()
