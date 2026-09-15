#!/usr/bin/env python3
"""
Railway Rover Multimodal Perception Node
- Camera-based anomaly detection: Cracks, Rust, Track Damage
- 3D LiDAR-based obstacle detection with calibrated spatial filtering
- Annotated visualization image publisher (/camera/annotated_image)
"""

import math
import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, PointCloud
from std_msgs.msg import Bool, Float32, String


class RailwayPerception:

    def __init__(self):
        rospy.init_node("railway_perception")
        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------
        self.crack_pub = rospy.Publisher("/crack_detected", Bool, queue_size=10)
        self.crack_score_pub = rospy.Publisher("/crack_score", Float32, queue_size=10)

        self.rust_pub = rospy.Publisher("/rust_detected", Bool, queue_size=10)
        self.rust_score_pub = rospy.Publisher("/rust_score", Float32, queue_size=10)

        self.anomaly_pub = rospy.Publisher("/anomaly_detected", Bool, queue_size=10)
        self.anomaly_info_pub = rospy.Publisher("/anomaly_info", String, queue_size=10)

        self.obstacle_pub = rospy.Publisher("/obstacle_detected", Bool, queue_size=10)
        self.obstacle_distance_pub = rospy.Publisher("/obstacle_distance", Float32, queue_size=10)
        self.obstacle_pos_pub = rospy.Publisher("/obstacle_position", PointStamped, queue_size=10)

        self.annotated_image_pub = rospy.Publisher("/camera/annotated_image", Image, queue_size=2)

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        rospy.Subscriber("/camera/image_raw", Image, self.camera_callback, queue_size=1)
        rospy.Subscriber("/lidar/points", PointCloud, self.lidar_callback, queue_size=1)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.latest_crack_score = 0.0
        self.latest_rust_score = 0.0
        self.latest_anomaly_detected = False
        self.latest_anomaly_type = "None"

        self.obstacle_state = False
        self.obstacle_distance = -1.0
        self.obstacle_hits = 0
        self.obstacle_misses = 0
        self.required_obstacle_hits = 2
        self.required_obstacle_misses = 4

        self.total_cracks_found = 0
        self.total_rust_found = 0
        self._crack_logged = False
        self._rust_logged = False

        rospy.loginfo("==============================================")
        rospy.loginfo(" Railway Rover Multimodal Perception Started")
        rospy.loginfo(" Camera raw: /camera/image_raw")
        rospy.loginfo(" Camera annotated: /camera/annotated_image")
        rospy.loginfo(" LiDAR: /lidar/points")
        rospy.loginfo(" Detection Topics: /crack_detected, /rust_detected, /obstacle_detected")
        rospy.loginfo("==============================================")

    # =============================================================
    # CAMERA CALLBACK & ANOMALY DETECTION
    # =============================================================
    def camera_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Camera conversion error: %s", str(e))
            return

        h, w = image.shape[:2]
        annotated = image.copy()

        # Define Railway Inspection Region of Interest (ROI)
        # Rails and track corridor ahead
        y_top = int(h * 0.40)
        y_bottom = int(h * 0.98)
        roi = image[y_top:y_bottom, :]

        # 1. Crack Detection (Dark linear features along rail lines)
        crack_score, crack_boxes = self.detect_cracks(roi, y_top)

        # 2. Rust Detection (Orange/reddish-brown oxidation features)
        rust_score, rust_boxes = self.detect_rust(roi, y_top)

        # 3. Overall Anomaly Assessment
        crack_detected = crack_score >= 0.20
        rust_detected = rust_score >= 0.22
        anomaly_detected = crack_detected or rust_detected

        anomaly_msgs = []
        if crack_detected:
            anomaly_msgs.append("CRACK (score: {:.2f})".format(crack_score))
            if not self._crack_logged:
                self.total_cracks_found += 1
                self._crack_logged = True
        else:
            self._crack_logged = False

        if rust_detected:
            anomaly_msgs.append("RUST (score: {:.2f})".format(rust_score))
            if not self._rust_logged:
                self.total_rust_found += 1
                self._rust_logged = True
        else:
            self._rust_logged = False

        anomaly_info_str = ", ".join(anomaly_msgs) if anomaly_msgs else "Clear"

        # Publish detection states
        self.crack_score_pub.publish(Float32(data=crack_score))
        self.crack_pub.publish(Bool(data=crack_detected))
        self.rust_score_pub.publish(Float32(data=rust_score))
        self.rust_pub.publish(Bool(data=rust_detected))
        self.anomaly_pub.publish(Bool(data=anomaly_detected))
        self.anomaly_info_pub.publish(String(data=anomaly_info_str))

        # Annotate Image
        self.annotate_visuals(annotated, crack_boxes, rust_boxes, crack_detected, rust_detected)

        # Publish Annotated Image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed publishing annotated image: %s", str(e))

    def detect_cracks(self, roi, y_offset):
        """Detect dark, elongated crack defect patterns along rails."""
        h_roi, w_roi = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold dark pixels
        dark = cv2.inRange(gray, 0, 70)

        # Morphological opening to remove isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

        # Corridor mask focusing on rails (left and right rail columns)
        corridor_mask = np.zeros_like(dark)
        pts = np.array([
            [int(w_roi * 0.10), 0],
            [int(w_roi * 0.90), 0],
            [int(w_roi * 0.98), h_roi],
            [int(w_roi * 0.02), h_roi]
        ], dtype=np.int32)
        cv2.fillPoly(corridor_mask, [pts], 255)
        dark = cv2.bitwise_and(dark, corridor_mask)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0.0
        boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 25 or area > 4000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            # Avoid full-width sleeper structures
            if w > w_roi * 0.35:
                continue

            aspect = float(w) / float(h)
            elongated = (aspect >= 1.6 or aspect <= 0.6)
            if not elongated:
                continue

            size_score = min(area / 400.0, 1.0)
            shape_score = min(max(aspect, 1.0 / aspect) / 4.0, 1.0)
            score = 0.5 * size_score + 0.5 * shape_score

            if score > 0.18:
                best_score = max(best_score, score)
                boxes.append((x, y + y_offset, w, h, score))

        return best_score, boxes

    def detect_rust(self, roi, y_offset):
        """Detect reddish-orange oxidation patches on rails via HSV segmentation."""
        h_roi, w_roi = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Rust color ranges in HSV (orange-brown rust)
        lower_rust1 = np.array([5, 80, 50])
        upper_rust1 = np.array([25, 255, 220])
        mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)

        # Also capture deeper red rust tones
        lower_rust2 = np.array([170, 70, 50])
        upper_rust2 = np.array([180, 255, 200])
        mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)

        rust_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)

        # Mask within rail corridor
        corridor_mask = np.zeros_like(rust_mask)
        pts = np.array([
            [int(w_roi * 0.10), 0],
            [int(w_roi * 0.90), 0],
            [int(w_roi * 0.98), h_roi],
            [int(w_roi * 0.02), h_roi]
        ], dtype=np.int32)
        cv2.fillPoly(corridor_mask, [pts], 255)
        rust_mask = cv2.bitwise_and(rust_mask, corridor_mask)

        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0.0
        boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > 6000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            aspect = float(w) / float(h)
            score = min(area / 350.0, 1.0)

            if score > 0.18:
                best_score = max(best_score, score)
                boxes.append((x, y + y_offset, w, h, score))

        return best_score, boxes

    def annotate_visuals(self, img, crack_boxes, rust_boxes, crack_detected, rust_detected):
        """Draw inspection overlays, bounding boxes, and HUD status banner."""
        h, w = img.shape[:2]

        # Draw Crack Bounding Boxes (Red)
        for (x, y, bw, bh, score) in crack_boxes:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 240), 2)
            label = "CRACK: {:.2f}".format(score)
            cv2.putText(img, label, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 240), 2)

        # Draw Rust Bounding Boxes (Orange/Gold)
        for (x, y, bw, bh, score) in rust_boxes:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 140, 255), 2)
            label = "RUST: {:.2f}".format(score)
            cv2.putText(img, label, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        # Top HUD Status Banner
        hud_h = 42
        cv2.rectangle(img, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.line(img, (0, hud_h), (w, hud_h), (80, 80, 80), 1)

        # HUD Status Texts
        status_text = "RAILWAY INSPECTION ROVER"
        cv2.putText(img, status_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Obstacle Status Indicator
        if self.obstacle_state:
            obs_str = "OBSTACLE: {:.2f}m [STOP]".format(self.obstacle_distance)
            obs_color = (0, 0, 255)
        else:
            obs_str = "PATH: CLEAR"
            obs_color = (0, 255, 0)
        cv2.putText(img, obs_str, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, obs_color, 1)

        # Anomaly Status Indicator
        if crack_detected and rust_detected:
            anom_str = "DEFECT: CRACK + RUST"
            anom_color = (0, 0, 255)
        elif crack_detected:
            anom_str = "DEFECT: CRACK DETECTED"
            anom_color = (0, 0, 255)
        elif rust_detected:
            anom_str = "DEFECT: RUST DETECTED"
            anom_color = (0, 140, 255)
        else:
            anom_str = "TRACK: HEALTHY"
            anom_color = (0, 255, 0)

        cv2.putText(img, anom_str, (w - 240, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, anom_color, 1)

    # =============================================================
    # LIDAR CALLBACK & SPATIAL OBSTACLE FILTERING
    # =============================================================
    def lidar_callback(self, msg):
        """
        Filter 3D LiDAR point cloud in lidar_link frame.
        LiDAR is at world height z = 0.86 m.
        Rail top is at z = -0.60 m in lidar frame.
        Ground is at z = -0.86 m.
        Obstacles on track are between z = -0.50 m and +0.20 m.
        Track width corridor: |y| <= 0.85 m.
        Forward range: 0.5 m <= x <= 7.5 m.
        """
        nearest_distance = float("inf")
        nearest_point = None
        valid_points = 0

        for point in msg.points:
            x = point.x
            y = point.y
            z = point.z

            # Forward inspection corridor
            if x < 0.85 or x > 7.5:
                continue

            # Track width corridor (between and just outside rails)
            if abs(y) > 0.65:
                continue

            # Height filter in lidar_link frame:
            # -0.50 avoids rails (z=-0.60) and sleepers (z=-0.67)
            # +0.25 avoids rover frame and high background
            if z < -0.50 or z > 0.25:
                continue

            dist = math.sqrt(x * x + y * y)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_point = point

            valid_points += 1

        # Obstacle confirmation logic (require consecutive hits)
        raw_obstacle = (valid_points >= 3)

        if raw_obstacle:
            self.obstacle_hits += 1
            self.obstacle_misses = 0
        else:
            self.obstacle_misses += 1
            self.obstacle_hits = 0

        if self.obstacle_hits >= self.required_obstacle_hits:
            self.obstacle_state = True
        elif self.obstacle_misses >= self.required_obstacle_misses:
            self.obstacle_state = False

        self.obstacle_pub.publish(Bool(data=self.obstacle_state))

        if self.obstacle_state and nearest_point is not None:
            self.obstacle_distance = nearest_distance
            self.obstacle_distance_pub.publish(Float32(data=nearest_distance))

            pos_msg = PointStamped()
            pos_msg.header = msg.header
            pos_msg.point.x = nearest_point.x
            pos_msg.point.y = nearest_point.y
            pos_msg.point.z = nearest_point.z
            self.obstacle_pos_pub.publish(pos_msg)

            rospy.loginfo_throttle(
                2.0,
                "[LiDAR] Obstacle at %.2fm ahead (pts=%d, y=%.2fm)",
                nearest_distance, valid_points, nearest_point.y
            )
        else:
            self.obstacle_distance = -1.0
            self.obstacle_distance_pub.publish(Float32(data=-1.0))


if __name__ == "__main__":
    try:
        RailwayPerception()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
