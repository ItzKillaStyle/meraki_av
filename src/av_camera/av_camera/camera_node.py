import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('device',         '/dev/video0')
        self.declare_parameter('width',          1920)
        self.declare_parameter('height',         1080)
        self.declare_parameter('fps',            15)
        self.declare_parameter('frame_id',       'camera_link')
        self.declare_parameter('compressed_quality', 80)   # JPEG quality 0-100

        self.device    = self.get_parameter('device').value
        self.width     = self.get_parameter('width').value
        self.height    = self.get_parameter('height').value
        self.fps       = self.get_parameter('fps').value
        self.frame_id  = self.get_parameter('frame_id').value
        self.jpeg_quality = self.get_parameter('compressed_quality').value

        self.bridge  = CvBridge()
        self.enabled = True
        self.cap     = None

        # ── Abrir cámara ──────────────────────────────────────────────────────
        self.open_camera()

        # ── Publishers ────────────────────────────────────────────────────────
        # Raw — usado por av_vision internamente
        self.pub_image = self.create_publisher(
            Image, '/camera/image_raw', 10)

        # Comprimido — para debug remoto desde PC con rqt_image_view
        self.pub_compressed = self.create_publisher(
            CompressedImage, '/camera/image_compressed', 10)

        # CameraInfo — necesario para robot_localization y cv_bridge calibrado
        self.pub_info = self.create_publisher(
            CameraInfo, '/camera/camera_info', 10)

        # ── Subscriber — habilitar/deshabilitar cámara en caliente ────────────
        self.create_subscription(
            Bool, '/camera/enable', self.cb_enable, 10)

        # ── Timer de captura ──────────────────────────────────────────────────
        self.create_timer(1.0 / self.fps, self.capture_cb)

        self.get_logger().info(
            f'Camera node iniciado — {self.device} '
            f'{self.width}x{self.height} @ {self.fps} fps'
        )

    # ── Abrir cámara UVC ──────────────────────────────────────────────────────

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(
                f'No se pudo abrir {self.device} — '
                f'verifica con: ls /dev/video*')
            self.cap = None
            return

        # Configura resolución y FPS en el driver V4L2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # Preferir MJPEG sobre YUYV — mucho menor ancho de banda USB
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Verifica resolución real obtenida
        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(
            f'Cámara abierta — resolución real: {real_w}x{real_h} @ {real_fps:.1f} fps'
        )

    # ── Callback enable/disable ───────────────────────────────────────────────

    def cb_enable(self, msg: Bool):
        self.enabled = msg.data
        self.get_logger().info(
            f'Cámara {"habilitada" if self.enabled else "deshabilitada"}')

    # ── Callback captura ──────────────────────────────────────────────────────

    def capture_cb(self):
        if not self.enabled or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn(
                'Frame no capturado — verifica conexión USB')
            return

        now = self.get_clock().now().to_msg()

        # ── Publica imagen raw ────────────────────────────────────────────────
        try:
            img_msg                  = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp     = now
            img_msg.header.frame_id  = self.frame_id
            self.pub_image.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error raw: {e}')
            return

        # ── Publica imagen comprimida JPEG ────────────────────────────────────
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            ret_enc, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret_enc:
                comp_msg                 = CompressedImage()
                comp_msg.header.stamp    = now
                comp_msg.header.frame_id = self.frame_id
                comp_msg.format          = 'jpeg'
                comp_msg.data            = buffer.tobytes()
                self.pub_compressed.publish(comp_msg)
        except Exception as e:
            self.get_logger().error(f'Error comprimiendo imagen: {e}')

        # ── Publica CameraInfo básico ─────────────────────────────────────────
        info                  = CameraInfo()
        info.header.stamp     = now
        info.header.frame_id  = self.frame_id
        info.width            = self.width
        info.height           = self.height
        # Matriz intrínseca identidad hasta calibrar la cámara
        info.k = [
            float(self.width),  0.0,                float(self.width)  / 2.0,
            0.0,                float(self.height), float(self.height) / 2.0,
            0.0,                0.0,                1.0
        ]
        info.distortion_model = 'plumb_bob'
        info.d                = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_info.publish(info)

    # ── Cierre limpio ─────────────────────────────────────────────────────────

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info('Cámara liberada')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()