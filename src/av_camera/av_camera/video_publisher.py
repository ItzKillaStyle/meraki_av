import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('fps',        15.0)
        self.declare_parameter('loop',       True)
        self.declare_parameter('frame_id',   'camera_link')

        self.video_path = self.get_parameter('video_path').value
        self.fps        = self.get_parameter('fps').value
        self.loop       = self.get_parameter('loop').value
        self.frame_id   = self.get_parameter('frame_id').value

        if not self.video_path:
            self.get_logger().error('Debes especificar video_path')
            return

        self.cap    = cv2.VideoCapture(self.video_path)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error(f'No se pudo abrir: {self.video_path}')
            return

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_logger().info(
            f'Video cargado: {self.video_path} | {total} frames | {self.fps} fps')

        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.create_timer(1.0 / self.fps, self.timer_cb)

    def timer_cb(self):
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
            else:
                self.get_logger().info('Video terminado')
                rclpy.shutdown()
                return

        msg                  = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
