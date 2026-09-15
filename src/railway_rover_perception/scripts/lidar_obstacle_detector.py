#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PointStamped
import math


class LidarObstacleDetector:
    def __init__(self):
        rospy.init_node("lidar_obstacle_detector")

        self.obstacle_pub = rospy.Publisher(
            "/obstacle_detected", Bool, queue_size=10
        )

        self.distance_pub = rospy.Publisher(
            "/obstacle_distance", Float32, queue_size=10
        )

        self.position_pub = rospy.Publisher(
            "/obstacle_position", PointStamped, queue_size=10
        )

        rospy.Subscriber(
            "/lidar/points",
            PointCloud,
            self.lidar_callback
        )

        self.min_distance = 0.5
        self.max_distance = 7.5
        self.min_height = -0.50
        self.max_height = 0.25

        rospy.loginfo("LiDAR obstacle detector started.")
        rospy.loginfo("Publishing obstacle detection, distance and position.")

    def lidar_callback(self, msg):
        nearest_point = None
        nearest_distance = float("inf")

        for point in msg.points:
            x = point.x
            y = point.y
            z = point.z

            distance = math.sqrt(x * x + y * y)

            if x < self.min_distance:
                continue

            if x > self.max_distance:
                continue

            if abs(y) > 1.0:
                continue

            if z < self.min_height or z > self.max_height:
                continue

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_point = point

        obstacle = nearest_point is not None

        self.obstacle_pub.publish(Bool(data=obstacle))

        if obstacle:
            self.distance_pub.publish(Float32(nearest_distance))

            position = PointStamped()
            position.header = msg.header
            position.point.x = nearest_point.x
            position.point.y = nearest_point.y
            position.point.z = nearest_point.z

            self.position_pub.publish(position)

            rospy.loginfo_throttle(
                2.0,
                "Obstacle detected: %.2f m ahead, y=%.2f m, z=%.2f m",
                nearest_distance,
                nearest_point.y,
                nearest_point.z
            )

        else:
            self.distance_pub.publish(Float32(-1.0))


if __name__ == "__main__":
    try:
        LidarObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
