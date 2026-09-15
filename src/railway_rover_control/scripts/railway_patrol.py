#!/usr/bin/env python3
"""
Railway Patrol Controller Node
- Maintains steady straight-line forward motion on the railway track
- Obstacle safety: smooth deceleration and complete stop before obstacles (stopping clearance >= 1.8m)
- Defect logging: tracks rover position and logs crack / rust inspection events
- Publishes patrol status to /patrol/status
"""

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String


class RailwayPatrol:

    def __init__(self):
        rospy.init_node("railway_patrol")

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.status_pub = rospy.Publisher("/patrol/status", String, queue_size=10)

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        rospy.Subscriber("/obstacle_detected", Bool, self.obstacle_cb, queue_size=1)
        rospy.Subscriber("/obstacle_distance", Float32, self.obstacle_dist_cb, queue_size=1)
        rospy.Subscriber("/crack_detected", Bool, self.crack_cb, queue_size=1)
        rospy.Subscriber("/rust_detected", Bool, self.rust_cb, queue_size=1)
        rospy.Subscriber("/anomaly_info", String, self.anomaly_info_cb, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=1)

        # ---------------------------------------------------------
        # Parameters & State
        # ---------------------------------------------------------
        self.patrol_speed = rospy.get_param("~patrol_speed", 0.60)
        self.stop_distance_threshold = rospy.get_param("~stop_distance", 2.20)
        self.slow_distance_threshold = rospy.get_param("~slow_distance", 4.00)

        self.current_speed = 0.0
        self.rover_x = -30.0

        self.obstacle_detected = False
        self.obstacle_distance = -1.0

        self.crack_detected = False
        self.rust_detected = False
        self.anomaly_info = "Clear"

        self._logged_cracks = []
        self._logged_rusts = []

        rospy.loginfo("==============================================")
        rospy.loginfo(" Railway Patrol Controller Initialized")
        rospy.loginfo(" Patrol speed: %.2f m/s", self.patrol_speed)
        rospy.loginfo(" Stop threshold: %.2f m", self.stop_distance_threshold)
        rospy.loginfo(" Status topic: /patrol/status")
        rospy.loginfo("==============================================")

    def obstacle_cb(self, msg):
        self.obstacle_detected = msg.data

    def obstacle_dist_cb(self, msg):
        self.obstacle_distance = msg.data

    def crack_cb(self, msg):
        self.crack_detected = msg.data
        if self.crack_detected:
            # Check if this crack was already logged within 2 meters
            already_logged = any(abs(self.rover_x - x) < 2.0 for x in self._logged_cracks)
            if not already_logged:
                self._logged_cracks.append(self.rover_x)
                rospy.logwarn(
                    "*** [INSPECTION LOG] CRACK detected at track position x = %.2f m ***",
                    self.rover_x
                )

    def rust_cb(self, msg):
        self.rust_detected = msg.data
        if self.rust_detected:
            already_logged = any(abs(self.rover_x - x) < 2.0 for x in self._logged_rusts)
            if not already_logged:
                self._logged_rusts.append(self.rover_x)
                rospy.logwarn(
                    "*** [INSPECTION LOG] RUST defect detected at track position x = %.2f m ***",
                    self.rover_x
                )

    def anomaly_info_cb(self, msg):
        self.anomaly_info = msg.data

    def odom_cb(self, msg):
        self.rover_x = msg.pose.pose.position.x

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz

        while not rospy.is_shutdown():
            cmd = Twist()
            status = "PATROLLING"

            # Determine Target Speed based on Safety Obstacle Detection
            if self.obstacle_detected and self.obstacle_distance > 0:
                if self.obstacle_distance <= self.stop_distance_threshold:
                    # Full Stop
                    target_speed = 0.0
                    status = "STOPPED_OBSTACLE"
                    rospy.logwarn_throttle(
                        2.0,
                        "[PATROL SAFETY] Obstacle %.2fm ahead -> EMERGENCY STOPPED at x=%.2fm",
                        self.obstacle_distance, self.rover_x
                    )
                elif self.obstacle_distance <= self.slow_distance_threshold:
                    # Controlled approach deceleration
                    ratio = (self.obstacle_distance - self.stop_distance_threshold) / (
                        self.slow_distance_threshold - self.stop_distance_threshold
                    )
                    target_speed = max(0.15, self.patrol_speed * ratio)
                    status = "APPROACHING_OBSTACLE"
                else:
                    target_speed = self.patrol_speed
            elif self.obstacle_detected:
                # Obstacle detected without valid distance: stop safely
                target_speed = 0.0
                status = "STOPPED_OBSTACLE"
            else:
                # Track is clear
                if self.crack_detected or self.rust_detected:
                    status = "INSPECTING_ANOMALY"
                    # Slight reduction in speed during inspection
                    target_speed = self.patrol_speed * 0.85
                else:
                    target_speed = self.patrol_speed
                    status = "PATROLLING"

            # Smooth Acceleration / Deceleration
            accel_limit = 0.05  # max speed change per step (1.0 m/s^2 at 20Hz)
            if target_speed > self.current_speed:
                self.current_speed = min(target_speed, self.current_speed + accel_limit)
            else:
                self.current_speed = max(target_speed, self.current_speed - accel_limit * 1.5)

            # Strictly forward straight-line motion (zero lateral drift)
            cmd.linear.x = self.current_speed
            cmd.linear.y = 0.0
            cmd.linear.z = 0.0
            cmd.angular.x = 0.0
            cmd.angular.y = 0.0
            cmd.angular.z = 0.0

            self.cmd_pub.publish(cmd)
            self.status_pub.publish(String(data=status))

            rate.sleep()


if __name__ == "__main__":
    try:
        patrol = RailwayPatrol()
        patrol.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.sleep(0.1)
        stop_cmd = Twist()
        stop_pub.publish(stop_cmd)
        rospy.sleep(0.1)
