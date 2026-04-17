#!/usr/bin/env python3
import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    if dot < 0:
        q2  = -q2
        dot = -dot
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    return (np.sin(theta_0 - theta) / sin_t0) * q1 + (np.sin(theta) / sin_t0) * q2


def array_to_quat_msg(q: np.ndarray) -> Quaternion:
    msg   = Quaternion()
    msg.w = float(q[0])
    msg.x = float(q[1])
    msg.y = float(q[2])
    msg.z = float(q[3])
    return msg


class LocalizationNode(Node):

    def __init__(self):
        super().__init__('localization_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('frame_id',       'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('imu_weight',     0.5)
        self.declare_parameter('publish_hz',     50.0)
        self.declare_parameter('alpha',          0.98)

        self.frame_id       = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.imu_weight     = self.get_parameter('imu_weight').value
        self.publish_hz     = self.get_parameter('publish_hz').value
        self.alpha          = self.get_parameter('alpha').value
        self.dt             = 1.0 / self.publish_hz

        # ── Estado interno ────────────────────────────────────────────────────
        self.imu_front_q  = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_rear_q   = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_front_ok = False
        self.imu_rear_ok  = False

        self.front_roll  = 0.0
        self.front_pitch = 0.0
        self.rear_roll   = 0.0
        self.rear_pitch  = 0.0

        self.gps_fix = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String, '/imu/front', self.cb_imu_front, 10)
        self.create_subscription(
            String, '/imu/rear',  self.cb_imu_rear,  10)
        self.create_subscription(
            NavSatFix, '/gps/fix', self.cb_gps, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_imu_fused = self.create_publisher(
            Imu, '/imu/fused', 10)
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/pose', 10)
        self.pub_odom = self.create_publisher(
            Odometry, '/localization/odom', 10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(self.dt, self.publish_cb)

        self.get_logger().info(
            f'Localization node iniciado — '
            f'fusión MPU6050 x2 | peso_frontal={self.imu_weight} '
            f'alpha={self.alpha}'
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rpy_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(roll  / 2)
        sr = math.sin(roll  / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw   / 2)
        sy = math.sin(yaw   / 2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return np.array([w, x, y, z])

    def _parse_imu(self, msg: String):
        try:
            d = json.loads(msg.data)
            return {k: float(d[k]) for k in ('ax', 'ay', 'az', 'gx', 'gy', 'gz')}
        except Exception as e:
            self.get_logger().warn(f'IMU parse error: {e}')
            return None

    def _complementary(self, roll, pitch, d):
        acel_roll  = math.atan2(d['ay'], math.sqrt(d['ax']**2 + d['az']**2))
        acel_pitch = math.atan2(d['ax'], math.sqrt(d['ay']**2 + d['az']**2))
        roll  = self.alpha * (roll  + math.radians(d['gx']) * self.dt) + (1 - self.alpha) * acel_roll
        pitch = self.alpha * (pitch + math.radians(d['gy']) * self.dt) + (1 - self.alpha) * acel_pitch
        return roll, pitch

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_imu_front(self, msg: String):
        d = self._parse_imu(msg)
        if d is None:
            return
        self.front_roll, self.front_pitch = self._complementary(
            self.front_roll, self.front_pitch, d)
        self.imu_front_q  = self._rpy_to_quat(self.front_roll, self.front_pitch, 0.0)
        self.imu_front_ok = True

    def cb_imu_rear(self, msg: String):
        d = self._parse_imu(msg)
        if d is None:
            return
        self.rear_roll, self.rear_pitch = self._complementary(
            self.rear_roll, self.rear_pitch, d)
        self.imu_rear_q  = self._rpy_to_quat(self.rear_roll, self.rear_pitch, 0.0)
        self.imu_rear_ok = True

    def cb_gps(self, msg: NavSatFix):
        self.gps_fix = msg

    # ── Publicación fusionada ─────────────────────────────────────────────────

    def publish_cb(self):
        now = self.get_clock().now().to_msg()

        if self.imu_front_ok and self.imu_rear_ok:
            q_fused = slerp(self.imu_front_q, self.imu_rear_q, 1.0 - self.imu_weight)
            source  = 'front+rear'
        elif self.imu_front_ok:
            q_fused = self.imu_front_q
            source  = 'front only'
        elif self.imu_rear_ok:
            q_fused = self.imu_rear_q
            source  = 'rear only'
        else:
            return

        q_fused = q_fused / np.linalg.norm(q_fused)

        # IMU fusionada → robot_localization
        imu_msg                 = Imu()
        imu_msg.header.stamp    = now
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.orientation     = array_to_quat_msg(q_fused)
        cov = 0.01
        imu_msg.orientation_covariance = [
            cov, 0.0, 0.0,
            0.0, cov, 0.0,
            0.0, 0.0, cov
        ]
        imu_msg.angular_velocity_covariance[0]    = -1.0
        imu_msg.linear_acceleration_covariance[0] = -1.0
        self.pub_imu_fused.publish(imu_msg)

        # Odometría
        odom                       = Odometry()
        odom.header.stamp          = now
        odom.header.frame_id       = self.frame_id
        odom.child_frame_id        = self.child_frame_id
        odom.pose.pose.orientation = array_to_quat_msg(q_fused)
        odom.pose.pose.position.x  = 0.0
        odom.pose.pose.position.y  = 0.0
        odom.pose.pose.position.z  = 0.0
        odom.pose.covariance[0]    = 0.1
        odom.pose.covariance[7]    = 0.1
        odom.pose.covariance[35]   = 0.05
        self.pub_odom.publish(odom)

        self.get_logger().debug(
            f'IMU fused [{source}] | '
            f'roll={math.degrees(self.front_roll):.1f}° '
            f'pitch={math.degrees(self.front_pitch):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()