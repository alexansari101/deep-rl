#!/usr/bin/env python

import sys
import rospy
from beginner_tutorials.srv import *
from std_msgs.msg import Int64

def add_two_ints_client(x, y):
    rospy.wait_for_service('add_two_ints')
    try:
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        resp1 = add_two_ints(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
        x=Int64(data=x)
        y=Int64(data=y)
        print (x.data)
        print (y.data)
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x.data, y.data)
    print "%s + %s = %s"%(x.data, y.data, add_two_ints_client(x, y).data)

