#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
import cv2
import os
import signal
from datetime import datetime


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.declare_parameter('output_dir', '/home/carrito/videos')
        self.declare_parameter('fps',         15.0)
        self.declare_parameter('width',       640)
        self.declare_parameter('height',      480)

        out_dir = self.get_parameter('output_dir').value
        fps     = self.get_parameter('fps').value
        width   = self.get_parameter('width').value
        height  = self.get_parameter('height').value

        os.makedirs(out_dir, exist_ok=True)

        self.create_subscription(Bool, '/recording/active',
                self.cb_recording, 10)

        self._filename = os.path.join(
            out_dir,
            datetime.now().strftime('meraki_%Y%m%d_%H%M%S.mp4')
        )

        fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer  = cv2.VideoWriter(
            self._filename, fourcc, fps, (width, height))
        self.bridge  = CvBridge()
        self._closed = False

        # Manejador de señales para cierre limpio
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.create_subscription(Image, '/camera/image_raw', self.cb_image, 10)
        self.get_logger().info(f'Grabando en: {self._filename}')

    def cb_image(self, msg: Image):
        if self._closed:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.writer.write(frame)
        except Exception as e:
            self.get_logger().error(f'Error frame: {e}')

    def _signal_handler(self, sig, frame):
        self._close()

    def cb_recording(self, msg: Bool):
        if not msg.data and not self._closed:
            self._close()
            self.get_logger().info('Grabación detenida por comando')
        elif msg.data and self._closed:
            # Reiniciar grabación con nuevo archivo
            self._start_new_recording()

    def _close(self):
        if not self._closed:
            self._closed = True
            self.writer.release()
            self.get_logger().info(f'Video guardado: {self._filename}')

    def destroy_node(self):
        self._close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
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