#!/usr/bin/env python3

import json

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Imu


class ImuParserNode(Node):

    def __init__(self):
        super().__init__('imu_parser_node')

        # ── Subscribers ─────────────────────────────────────────────

        self.create_subscription(
            String,
            '/imu/front',
            self.cb_front,
            10
        )

        self.create_subscription(
            String,
            '/imu/rear',
            self.cb_rear,
            10
        )

        # ── Publishers ──────────────────────────────────────────────

        self.pub_front = self.create_publisher(
            Imu,
            '/imu/front/data',
            10
        )

        self.pub_rear = self.create_publisher(
            Imu,
            '/imu/rear/data',
            10
        )

        self.get_logger().info('IMU Parser Node iniciado')

    # ───────────────────────────────────────────────────────────────

    def cb_front(self, msg):
        self._process_imu(
            msg.data,
            self.pub_front,
            'imu_front'
        )

    def cb_rear(self, msg):
        self._process_imu(
            msg.data,
            self.pub_rear,
            'imu_rear'
        )

    # ───────────────────────────────────────────────────────────────

    def _process_imu(self, data, publisher, frame_id):

        try:
            obj = json.loads(data)

        except Exception as e:
            self.get_logger().warn(
                f'JSON inválido IMU: {e}'
            )
            return

        imu = Imu()

        imu.header.stamp = (
            self.get_clock().now().to_msg()
        )

        imu.header.frame_id = frame_id

        # ── Gyroscope ───────────────────────────────────────────

        imu.angular_velocity.x = float(
            obj.get('gx', 0.0)
        )

        imu.angular_velocity.y = float(
            obj.get('gy', 0.0)
        )

        imu.angular_velocity.z = float(
            obj.get('gz', 0.0)
        )

        # ── Accelerometer ───────────────────────────────────────

        imu.linear_acceleration.x = float(
            obj.get('ax', 0.0)
        )

        imu.linear_acceleration.y = float(
            obj.get('ay', 0.0)
        )

        imu.linear_acceleration.z = float(
            obj.get('az', 0.0)
        )

        # ── Covarianzas simples ─────────────────────────────────

        imu.orientation_covariance[0] = -1.0

        imu.angular_velocity_covariance[0] = 0.02
        imu.angular_velocity_covariance[4] = 0.02
        imu.angular_velocity_covariance[8] = 0.02

        imu.linear_acceleration_covariance[0] = 0.04
        imu.linear_acceleration_covariance[4] = 0.04
        imu.linear_acceleration_covariance[8] = 0.04

        publisher.publish(imu)


def main(args=None):

    rclpy.init(args=args)

    node = ImuParserNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()