#!/usr/bin/env python3

import math
import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud
from std_msgs.msg import Bool, Float32


class RailwayPerception:

    def __init__(self):
        rospy.init_node("railway_perception")

        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------

        self.crack_pub = rospy.Publisher(
            "/crack_detected",
            Bool,
            queue_size=10
        )

        self.crack_score_pub = rospy.Publisher(
            "/crack_score",
            Float32,
            queue_size=10
        )

        self.obstacle_pub = rospy.Publisher(
            "/obstacle_detected",
            Bool,
            queue_size=10
        )

        self.obstacle_distance_pub = rospy.Publisher(
            "/obstacle_distance",
            Float32,
            queue_size=10
        )

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        rospy.Subscriber(
            "/camera/image_raw",
            Image,
            self.camera_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/lidar/points",
            PointCloud,
            self.lidar_callback,
            queue_size=1
        )

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------

        self.latest_crack_score = 0.0

        self.obstacle_state = False
        self.obstacle_distance = -1.0

        # Require several consecutive LiDAR detections.
        self.obstacle_hits = 0
        self.obstacle_misses = 0

        self.required_obstacle_hits = 3
        self.required_obstacle_misses = 5

        rospy.loginfo("==============================================")
        rospy.loginfo(" Railway Rover Multimodal Perception Started")
        rospy.loginfo(" RGB camera : /camera/image_raw")
        rospy.loginfo(" 3D LiDAR   : /lidar/points")
        rospy.loginfo(" Crack      : /crack_detected")
        rospy.loginfo(" Obstacle   : /obstacle_detected")
        rospy.loginfo("==============================================")

    # =============================================================
    # CAMERA
    # =============================================================

    def camera_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                "Camera conversion failed: %s",
                str(e)
            )
            return

        score = self.detect_crack(image)

        self.latest_crack_score = score

        self.crack_score_pub.publish(
            Float32(data=score)
        )

        detected = score >= 0.18

        self.crack_pub.publish(
            Bool(data=detected)
        )

        if detected:
            rospy.loginfo_throttle(
                2.0,
                "CAMERA: possible crack detected, score=%.2f",
                score
            )

    # =============================================================
    # DEMO CAMERA CRACK DETECTOR
    # =============================================================

    def detect_crack(self, image):

        height, width = image.shape[:2]

        # Ignore sky and distant scene.
        y1 = int(height * 0.48)
        y2 = int(height * 0.94)

        roi = image[y1:y2, :]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # Dark structures are potential crack marks.
        dark = cv2.inRange(
            gray,
            0,
            65
        )

        # Remove isolated noise.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            kernel
        )

        # Only inspect the railway corridor.
        mask = np.zeros_like(dark)

        corridor = [
            (
                int(width * 0.20),
                0
            ),
            (
                int(width * 0.80),
                0
            ),
            (
                int(width * 1.00),
                int(roi.shape[0] * 0.98)
            ),
            (
                0,
                int(roi.shape[0] * 0.98)
            )
        ]

        cv2.fillPoly(
            mask,
            [__import__("numpy").array(corridor)],
            255
        )

        dark = cv2.bitwise_and(
            dark,
            mask
        )

        # Find candidate dark structures.
        contours, _ = cv2.findContours(
            dark,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_score = 0.0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 20 or area > 3000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            aspect = float(w) / float(h)

            # Crack-like structures tend to be elongated.
            elongated = (
                aspect >= 2.0 or
                aspect <= 0.5
            )

            if not elongated:
                continue

            # Avoid very large horizontal sleeper structures.
            if w > width * 0.30:
                continue

            # Candidate quality.
            size_score = min(
                area / 500.0,
                1.0
            )

            shape_score = min(
                max(aspect, 1.0 / aspect) / 5.0,
                1.0
            )

            candidate_score = (
                0.55 * size_score +
                0.45 * shape_score
            )

            best_score = max(
                best_score,
                candidate_score
            )

        return best_score

    # =============================================================
    # LIDAR
    # =============================================================

    def lidar_callback(self, msg):

        nearest_distance = float("inf")
        valid_points = 0

        for point in msg.points:

            x = point.x
            y = point.y
            z = point.z

            # Forward region only.
            if x < 0.6 or x > 3.0:
                continue

            # Keep points near the track center.
            if abs(y) > 1.0:
                continue

            # Ignore the rail/sleeper plane and high background.
            if z < 0.35 or z > 1.5:
                continue

            distance = math.sqrt(
                x * x + y * y
            )

            if distance < nearest_distance:
                nearest_distance = distance

            valid_points += 1

        # Require multiple points before declaring a real obstacle.
        raw_obstacle = valid_points >= 4

        if raw_obstacle:
            self.obstacle_hits += 1
            self.obstacle_misses = 0
        else:
            self.obstacle_misses += 1
            self.obstacle_hits = 0

        if self.obstacle_hits >= self.required_obstacle_hits:
            self.obstacle_state = True

        if self.obstacle_misses >= self.required_obstacle_misses:
            self.obstacle_state = False

        self.obstacle_pub.publish(
            Bool(data=self.obstacle_state)
        )

        if self.obstacle_state:
            self.obstacle_distance = nearest_distance

            self.obstacle_distance_pub.publish(
                Float32(data=nearest_distance)
            )

            rospy.loginfo_throttle(
                2.0,
                "LIDAR: obstacle %.2f m ahead (%d points)",
                nearest_distance,
                valid_points
            )

        else:
            self.obstacle_distance_pub.publish(
                Float32(data=-1.0)
            )


if __name__ == "__main__":

    try:
        RailwayPerception()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
