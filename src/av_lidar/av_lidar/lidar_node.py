#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('frame_id',              'lidar_link')
        self.declare_parameter('range_min',              0.15)
        self.declare_parameter('range_max',              8.0)
        self.declare_parameter('angle_offset_deg',       0.0)   # ← 180.0 si está al revés

        self.frame_id     = self.get_parameter('frame_id').value
        self.range_min    = self.get_parameter('range_min').value
        self.range_max    = self.get_parameter('range_max').value
        self.angle_offset = np.deg2rad(
            self.get_parameter('angle_offset_deg').value)

        self.sub = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, 10)
        self.pub = self.create_publisher(
            LaserScan, '/lidar/scan', 10)

        self.get_logger().info(
            f'Lidar node | /scan → /lidar/scan | '
            f'rango [{self.range_min}, {self.range_max}] m | '
            f'offset={np.degrees(self.angle_offset):.1f}°'
        )

    def cb_scan(self, msg: LaserScan):
        msg.header.frame_id = self.frame_id
        msg.range_min       = self.range_min
        msg.range_max       = self.range_max

        # Aplicar offset de rotación si el LiDAR está montado rotado
        if self.angle_offset != 0.0:
            msg.angle_min += self.angle_offset
            msg.angle_max += self.angle_offset

            # Reordenar el array de rangos para que 0° quede al frente
            n      = len(msg.ranges)
            shift  = int(round(self.angle_offset / msg.angle_increment))
            shift  = shift % n  # asegurar que esté en rango
            ranges = list(msg.ranges)
            ranges = ranges[shift:] + ranges[:shift]
            msg.ranges = ranges

            # Mismo reordenamiento para intensidades si existen
            if msg.intensities:
                intens = list(msg.intensities)
                intens = intens[shift:] + intens[:shift]
                msg.intensities = intens

            # Normalizar ángulos a [-π, π]
            while msg.angle_min >  np.pi: msg.angle_min -= 2 * np.pi
            while msg.angle_min < -np.pi: msg.angle_min += 2 * np.pi
            while msg.angle_max >  np.pi: msg.angle_max -= 2 * np.pi
            while msg.angle_max < -np.pi: msg.angle_max += 2 * np.pi

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()