#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.declare_parameter('output_dir', '/home/carrito/videos')
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        out_dir = self.get_parameter('output_dir').value
        fps     = self.get_parameter('fps').value
        width   = self.get_parameter('width').value
        height  = self.get_parameter('height').value

        os.makedirs(out_dir, exist_ok=True)

        filename = os.path.join(
            out_dir,
            datetime.now().strftime('meraki_%Y%m%d_%H%M%S.mp4')
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        self.bridge = CvBridge()

        self.create_subscription(Image, '/camera/image_raw', self.cb_image, 10)
        self.get_logger().info(f'Grabando en: {filename}')

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.writer.write(frame)
        except Exception as e:
            self.get_logger().error(f'Error frame: {e}')

    def destroy_node(self):
        self.writer.release()
        self.get_logger().info('Video guardado')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()