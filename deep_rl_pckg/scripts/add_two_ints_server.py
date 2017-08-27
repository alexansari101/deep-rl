#!/usr/bin/env python

from beginner_tutorials.srv import *
import rospy
from std_msgs.msg import Int64

def handle_add_two_ints(req):
    print "Returning [%s + %s = %s]"%(req.a.data, req.b.data, (req.a.data + req.b.data))
    
    return AddTwoIntsResponse(Int64(data=(req.a.data + req.b.data)))

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()