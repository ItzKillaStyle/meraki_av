import rclpy
from rclpy.node import Node


class Diagnostics_node(Node):

    def __init__(self):
        super().__init__('diagnostics_node')
        self.get_logger().info('diagnostics_node iniciado')


def main(args=None):
    rclpy.init(args=args)
    node = Diagnostics_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
