#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class RailwayPatrol:
    def __init__(self):
        rospy.init_node("railway_patrol")

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=10
        )

        rospy.Subscriber(
            "/obstacle_detected",
            Bool,
            self.obstacle_callback
        )

        self.obstacle_detected = False
        self.forward_speed = 1.00

        rospy.loginfo("Railway patrol controller started.")
        rospy.loginfo("Obstacle-aware rail patrol enabled.")

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data

    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():

            cmd = Twist()

            if self.obstacle_detected:
                # Stop immediately when LiDAR detects an obstacle.
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                rospy.logwarn_throttle(
                    2.0,
                    "Obstacle detected - rover stopped."
                )
            else:
                # Move straight along the railway.
                cmd.linear.x = self.forward_speed
                cmd.angular.z = 0.0

            self.cmd_pub.publish(cmd)

            rate.sleep()


if __name__ == "__main__":
    try:
        controller = RailwayPatrol()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.sleep(0.2)
        stop_cmd = Twist()
        stop_pub.publish(stop_cmd)
        rospy.sleep(0.1)
