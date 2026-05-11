#!/usr/bin/env python3
"""
MERAKI — Traffic Sign Node
Detecta señales de tránsito usando YOLOv8n y publica en /perception/traffic_sign
También detecta peatones y vehículos usando el modelo base COCO (yolov8n.pt)
"""
import cv2
import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from av_interfaces.msg import TrafficSign
from std_msgs.msg import String

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

# ── Mapeo clase → (id TrafficSign, sign_type) ─────────────────────────────────
# sign_type debe coincidir EXACTAMENTE con SignName en behavior_node.py
CLASS_TO_SIGN = {
    "Ceda el paso":              (TrafficSign.GIVE_WAY,            "Ceda el paso"),
    "Cruce peatonal":            (TrafficSign.NO_SIGN,             "Cruce peatonal"),
    "Parada de bus":             (TrafficSign.NO_SIGN,             "Parada de bus"),
    "Pare":                      (TrafficSign.STOP,                "Pare"),
    "Prohibido el giro en U":    (TrafficSign.NO_SIGN,             "Prohibido el giro en U"),
    "Prohibido parquear":        (TrafficSign.NO_SIGN,             "Prohibido parquear"),
    "Proximidad cruce peatonal": (TrafficSign.NO_SIGN,             "Proximidad cruce peatonal"),
    "Semaforo peatonal rojo":    (TrafficSign.TRAFFIC_LIGHT_RED,   "Semaforo peatonal rojo"),
    "Semaforo peatonal verde":   (TrafficSign.TRAFFIC_LIGHT_GREEN, "Semaforo peatonal verde"),
    "Velocidad maxima 30km":     (TrafficSign.SPEED_LIMIT,         "Velocidad maxima 30km"),
}

# Clases COCO relevantes
COCO_RELEVANT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class TrafficSignNode(Node):

    def __init__(self):
        super().__init__('traffic_sign_node')

        if not ULTRALYTICS_OK:
            self.get_logger().error('ultralytics no instalado — pip install ultralytics')
            return

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',      '/home/carrito/best.pt')
        self.declare_parameter('conf_threshold',   0.5)
        self.declare_parameter('device',           'cpu')
        self.declare_parameter('imgsz',            320)
        self.declare_parameter('detect_vehicles',  True)
        self.declare_parameter('debug',            False)
        self.declare_parameter('frame_id',         'camera_link')

        model_path    = self.get_parameter('model_path').value
        self.conf     = self.get_parameter('conf_threshold').value
        self.device   = self.get_parameter('device').value
        self.imgsz    = self.get_parameter('imgsz').value
        self.det_veh  = self.get_parameter('detect_vehicles').value
        self.debug    = self.get_parameter('debug').value
        self.frame_id = self.get_parameter('frame_id').value

        # ── Modelos ───────────────────────────────────────────────────────────
        try:
            self.model_signs = YOLO(model_path)
            self.get_logger().info(f'Modelo señales: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Error cargando modelo: {e}')
            self.model_signs = None

        self.model_coco = None
        if self.det_veh:
            try:
                self.model_coco = YOLO('yolov8n.pt')
                self.get_logger().info('Modelo COCO cargado')
            except Exception as e:
                self.get_logger().warn(f'No se pudo cargar COCO: {e}')

        self.bridge = CvBridge()
        self.last_result = None  # cache del último resultado para debug

        # ── Subscribers / Publishers ──────────────────────────────────────────
        self.create_subscription(Image, '/camera/image_raw', self.cb_image, 10)
        self.pub_sign    = self.create_publisher(TrafficSign, '/perception/traffic_sign', 10)
        self.pub_objects = self.create_publisher(String,      '/perception/objects',       10)
        self.pub_debug   = self.create_publisher(Image,       '/vision/signs_debug',       10)

        self.get_logger().info(
            f'Traffic Sign Node | conf={self.conf} device={self.device} imgsz={self.imgsz}')

    # ── Callback ──────────────────────────────────────────────────────────────

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        now = self.get_clock().now().to_msg()
        if self.model_signs:
            self._detect_signs(frame, now)
        if self.model_coco:
            self._detect_objects(frame, now)
        if self.debug:
            self._publish_debug(frame, now)

    def _detect_signs(self, frame, stamp):
        try:
            results = self.model_signs.predict(
                source=frame, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False)
        except Exception as e:
            self.get_logger().error(f'Inferencia señales: {e}')
            return

        self.last_result = results[0]  # cache para debug — evita segunda inferencia
        boxes = results[0].boxes
        if len(boxes) == 0:
            self._publish_sign(stamp, TrafficSign.NO_SIGN, 'none', 0.0)
            return

        # Detección con mayor confianza
        best_conf, best_class = 0.0, None
        for box in boxes:
            c = float(box.conf)
            cls = self.model_signs.names[int(box.cls)]
            if c > best_conf:
                best_conf, best_class = c, cls

        if best_class and best_class in CLASS_TO_SIGN:
            sign_id, sign_type = CLASS_TO_SIGN[best_class]
            self._publish_sign(stamp, sign_id, sign_type, best_conf)
            self.get_logger().debug(f'Señal: {sign_type} conf={best_conf:.2f}')

    def _detect_objects(self, frame, stamp):
        try:
            results = self.model_coco.predict(
                source=frame, conf=self.conf, imgsz=self.imgsz,
                device=self.device, classes=list(COCO_RELEVANT.keys()), verbose=False)
        except Exception as e:
            self.get_logger().error(f'Inferencia COCO: {e}')
            return

        objects = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id in COCO_RELEVANT:
                objects.append({
                    'type': COCO_RELEVANT[cls_id],
                    'conf': round(float(box.conf), 3),
                    'x':   round(float(box.xywhn[0][0]), 3),
                    'y':   round(float(box.xywhn[0][1]), 3),
                    'w':   round(float(box.xywhn[0][2]), 3),
                    'h':   round(float(box.xywhn[0][3]), 3),
                })
        if objects:
            msg = String(); msg.data = json.dumps(objects)
            self.pub_objects.publish(msg)
            self.get_logger().debug(f'Objetos: {len(objects)}')

    def _publish_sign(self, stamp, sign_id: int, sign_type: str, conf: float):
        msg = TrafficSign()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.frame_id
        msg.id         = sign_id
        msg.sign_type  = sign_type
        msg.confidence = conf
        msg.distance   = 0.0
        self.pub_sign.publish(msg)

    def _publish_debug(self, frame, stamp):
        try:
            # Reutiliza el resultado cacheado de _detect_signs — sin segunda inferencia
            if self.last_result is not None:
                annotated = self.last_result.plot()
            else:
                annotated = frame
            debug_msg                 = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_msg.header.stamp    = stamp
            debug_msg.header.frame_id = self.frame_id
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()