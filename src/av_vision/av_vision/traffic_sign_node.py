#!/usr/bin/env python3
"""
MERAKI — Traffic Sign Node
Detecta señales de tránsito usando YOLOv8n ONNX y publica en /perception/traffic_sign
También detecta peatones y vehículos usando el modelo base COCO (yolov8n.pt)
"""

import cv2
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


# ── Mapeo clases del modelo → ID del mensaje TrafficSign ─────────────────────
# Clases del modelo entrenado (orden del data.yaml)
SIGN_CLASSES = [
    "Ceda el paso",              # 0
    "Cruce peatonal",            # 1
    "Parada de bus",             # 2
    "Pare",                      # 3
    "Prohibido el giro en U",    # 4
    "Prohibido parquear",        # 5
    "Proximidad cruce peatonal", # 6
    "Semaforo peatonal rojo",    # 7
    "Semaforo peatonal verde",   # 8
    "Velocidad maxima 30km",     # 9
]

# Mapeo clase → (id TrafficSign, sign_type string)
CLASS_TO_SIGN = {
    "Ceda el paso":              (TrafficSign.GIVE_WAY,           "Ceda el paso"),
    "Cruce peatonal":            (TrafficSign.NO_SIGN,            "Cruce peatonal"),
    "Parada de bus":             (TrafficSign.NO_SIGN,            "Parada de bus"),
    "Pare":                      (TrafficSign.STOP,               "Pare"),
    "Prohibido el giro en U":    (TrafficSign.NO_SIGN,            "Prohibido giro U"),
    "Prohibido parquear":        (TrafficSign.NO_SIGN,            "Prohibido parquear"),
    "Proximidad cruce peatonal": (TrafficSign.NO_SIGN,            "Proximidad cruce"),
    "Semaforo peatonal rojo":    (TrafficSign.TRAFFIC_LIGHT_RED,  "Semaforo rojo"),
    "Semaforo peatonal verde":   (TrafficSign.TRAFFIC_LIGHT_GREEN,"Semaforo verde"),
    "Velocidad maxima 30km":     (TrafficSign.SPEED_LIMIT,        "Velocidad 30km"),
}

# Clases COCO relevantes para MERAKI
COCO_RELEVANT = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}


class TrafficSignNode(Node):

    def __init__(self):
        super().__init__('traffic_sign_node')

        if not ULTRALYTICS_OK:
            self.get_logger().error('ultralytics no instalado — pip install ultralytics')
            return

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',      '/home/victor/meraki_web/training/meraki_vision/senales_v2/weights/best.pt')
        self.declare_parameter('conf_threshold',   0.5)
        self.declare_parameter('device',           'cpu')   # cpu para RPi5
        self.declare_parameter('imgsz',            320)     # reducido para RPi5
        self.declare_parameter('detect_vehicles',  True)
        self.declare_parameter('debug',            False)
        self.declare_parameter('frame_id',         'camera_link')

        model_path       = self.get_parameter('model_path').value
        self.conf        = self.get_parameter('conf_threshold').value
        self.device      = self.get_parameter('device').value
        self.imgsz       = self.get_parameter('imgsz').value
        self.det_veh     = self.get_parameter('detect_vehicles').value
        self.debug       = self.get_parameter('debug').value
        self.frame_id    = self.get_parameter('frame_id').value

        # ── Cargar modelo de señales ──────────────────────────────────────────
        try:
            self.model_signs = YOLO(model_path)
            self.get_logger().info(f'Modelo señales cargado: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Error cargando modelo: {e}')
            self.model_signs = None

        # ── Cargar modelo COCO para peatones y vehículos ──────────────────────
        if self.det_veh:
            try:
                self.model_coco = YOLO('yolov8n.pt')
                self.get_logger().info('Modelo COCO cargado para peatones/vehículos')
            except Exception as e:
                self.get_logger().warn(f'No se pudo cargar modelo COCO: {e}')
                self.model_coco = None
        else:
            self.model_coco = None

        self.bridge = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/camera/image_raw', self.cb_image, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_sign = self.create_publisher(
            TrafficSign, '/perception/traffic_sign', 10)

        # Objetos detectados (peatones/vehículos) como string JSON
        self.pub_objects = self.create_publisher(
            String, '/perception/objects', 10)

        # Imagen debug
        self.pub_debug = self.create_publisher(
            Image, '/vision/signs_debug', 10)

        self.get_logger().info(
            f'Traffic Sign Node iniciado | '
            f'conf={self.conf} device={self.device} imgsz={self.imgsz}'
        )

    # ── Callback principal ────────────────────────────────────────────────────

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        now = self.get_clock().now().to_msg()

        # ── Detección de señales ──────────────────────────────────────────────
        if self.model_signs:
            self._detect_signs(frame, now)

        # ── Detección de peatones y vehículos ─────────────────────────────────
        if self.model_coco:
            self._detect_objects(frame, now)

        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
            self._publish_debug(frame, now)

    def _detect_signs(self, frame, stamp):
        try:
            results = self.model_signs.predict(
                source  = frame,
                conf    = self.conf,
                imgsz   = self.imgsz,
                device  = self.device,
                verbose = False,
            )
        except Exception as e:
            self.get_logger().error(f'Error inferencia señales: {e}')
            return

        result = results[0]

        if len(result.boxes) == 0:
            # Publica NO_SIGN
            self._publish_sign(
                stamp, TrafficSign.NO_SIGN, "none", 0.0)
            return

        # Toma la detección con mayor confianza
        best_conf  = 0.0
        best_class = None
        for box in result.boxes:
            conf  = float(box.conf)
            clase = self.model_signs.names[int(box.cls)]
            if conf > best_conf:
                best_conf  = conf
                best_class = clase

        if best_class and best_class in CLASS_TO_SIGN:
            sign_id, sign_type = CLASS_TO_SIGN[best_class]
            self._publish_sign(stamp, sign_id, sign_type, best_conf)
            self.get_logger().debug(
                f'Señal: {sign_type} | conf={best_conf:.2f}')

    def _detect_objects(self, frame, stamp):
        try:
            results = self.model_coco.predict(
                source  = frame,
                conf    = self.conf,
                imgsz   = self.imgsz,
                device  = self.device,
                classes = list(COCO_RELEVANT.keys()),
                verbose = False,
            )
        except Exception as e:
            self.get_logger().error(f'Error inferencia COCO: {e}')
            return

        result  = results[0]
        objects = []

        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id in COCO_RELEVANT:
                objects.append({
                    'type': COCO_RELEVANT[cls_id],
                    'conf': round(float(box.conf), 3),
                    'x':    round(float(box.xywhn[0][0]), 3),  # centro norm
                    'y':    round(float(box.xywhn[0][1]), 3),
                    'w':    round(float(box.xywhn[0][2]), 3),
                    'h':    round(float(box.xywhn[0][3]), 3),
                })

        if objects:
            import json
            msg      = String()
            msg.data = json.dumps(objects)
            self.pub_objects.publish(msg)
            self.get_logger().debug(f'Objetos: {len(objects)} detectados')

    def _publish_sign(self, stamp, sign_id: int, sign_type: str, conf: float):
        msg            = TrafficSign()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.frame_id
        msg.id         = sign_id
        msg.sign_type  = sign_type
        msg.confidence = conf
        msg.distance   = 0.0  # sin estimación de distancia por ahora
        self.pub_sign.publish(msg)

    def _publish_debug(self, frame, stamp):
        try:
            if self.model_signs:
                results  = self.model_signs.predict(
                    frame, conf=self.conf, imgsz=self.imgsz,
                    device=self.device, verbose=False)
                annotated = results[0].plot()
            else:
                annotated = frame
            debug_msg        = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_msg.header.stamp    = stamp
            debug_msg.header.frame_id = self.frame_id
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error debug: {e}')


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