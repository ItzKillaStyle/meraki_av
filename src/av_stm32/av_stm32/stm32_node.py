import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class Stm32Node(Node):
    """
    Con micro-ROS la STM32 es un nodo ROS2 nativo.
    Este nodo solo actúa como watchdog y monitor —
    los topics /imu/front, /imu/rear, /ultrasonic/*
    los publica directamente la STM32 via micro-ROS agent.
    Este nodo supervisa que la STM32 siga viva.
    """

    def __init__(self):
        super().__init__('stm32_node')

        self.declare_parameter('watchdog_s', 1.0)
        self.watchdog_s = self.get_parameter('watchdog_s').value

        # Monitorea que llegan datos de la STM32
        self.last_imu_time  = self.get_clock().now()
        self.stm32_alive    = False

        self.create_subscription(
            String,
            '/imu/front',
            self.cb_imu_alive, 10)

        self.pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)

        self.create_timer(self.watchdog_s, self.watchdog_cb)

        self.get_logger().info(
            'STM32 node iniciado — micro-ROS agent maneja la comunicación\n'
            'Lanza el agente con:\n'
            '  ros2 run micro_ros_agent micro_ros_agent serial '
            '--dev /dev/ttyAMA0 -b 115200'
        )

    def cb_imu_alive(self, msg):
        self.last_imu_time = self.get_clock().now()
        if not self.stm32_alive:
            self.stm32_alive = True
            self.get_logger().info('STM32 conectada via micro-ROS')

    def watchdog_cb(self):
        elapsed = (self.get_clock().now() - self.last_imu_time).nanoseconds / 1e9
        if self.stm32_alive and elapsed > self.watchdog_s * 3:
            self.get_logger().warn('STM32 sin respuesta — activando emergency stop')
            self.stm32_alive = False
            estop      = Bool()
            estop.data = True
            self.pub_estop.publish(estop)


def main(args=None):
    rclpy.init(args=args)
    node = Stm32Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
