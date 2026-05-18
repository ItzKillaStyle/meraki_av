#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
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

        self.out_dir = self.get_parameter('output_dir').value
        self.fps     = self.get_parameter('fps').value
        self.width   = self.get_parameter('width').value
        self.height  = self.get_parameter('height').value

        os.makedirs(self.out_dir, exist_ok=True)

        self.bridge = CvBridge()

        self.writer = None
        self.recording = False

        self.create_subscription(
            Bool,
            '/recording/active',
            self.cb_recording,
            10
        )

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.cb_image,
            10
        )

        self.get_logger().info('Video recorder listo')

    def _start_new_recording(self):

        filename = os.path.join(
            self.out_dir,
            datetime.now().strftime('meraki_%Y%m%d_%H%M%S.mp4')
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        self.recording = True
        self.current_file = filename

        self.get_logger().info(f'Grabando: {filename}')

    def _stop_recording(self):

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.recording = False

        self.get_logger().info(
            f'Video guardado: {self.current_file}'
        )

    def cb_recording(self, msg: Bool):

        # START
        if msg.data and not self.recording:
            self._start_new_recording()

        # STOP
        elif not msg.data and self.recording:
            self._stop_recording()

    def cb_image(self, msg: Image):

        if not self.recording:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self.writer.write(frame)

        except Exception as e:
            self.get_logger().error(f'Error frame: {e}')

    def destroy_node(self):

        if self.recording:
            self._stop_recording()

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