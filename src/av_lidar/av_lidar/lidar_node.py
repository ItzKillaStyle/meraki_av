import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarNode(Node):

    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('frame_id',    'lidar_link')
        self.declare_parameter('range_min',   0.15)
        self.declare_parameter('range_max',   8.0)

        self.frame_id  = self.get_parameter('frame_id').value
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value

        # rplidar_ros publica en /scan — hacemos relay con filtrado
        self.sub = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, 10)

        self.pub = self.create_publisher(
            LaserScan, '/lidar/scan', 10)

        self.get_logger().info(
            f'Lidar node iniciado — relay /scan → /lidar/scan '
            f'rango [{self.range_min}, {self.range_max}] m'
        )

    def cb_scan(self, msg: LaserScan):
        # Corrige frame_id y aplica filtro de rango
        msg.header.frame_id = self.frame_id
        msg.range_min       = self.range_min
        msg.range_max       = self.range_max
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
