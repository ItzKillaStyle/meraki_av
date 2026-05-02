import math

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image

from av_interfaces.msg import LaneDetection

from av_vision.procesamiento_lineas import detectar_y_analizar_lineas


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ── Parámetros configurables desde YAML ──────────────────────────────
        self.declare_parameter('debug',           False)
        self.declare_parameter('alpha',           0.7)
        self.declare_parameter('umbral_calidad',  0.3)
        self.declare_parameter('frame_id',        'camera_link')

        self.debug          = self.get_parameter('debug').value
        self.alpha          = self.get_parameter('alpha').value
        self.umbral_calidad = self.get_parameter('umbral_calidad').value
        self.frame_id       = self.get_parameter('frame_id').value

        self.bridge = CvBridge()

        # ── Estado del filtro exponencial ────────────────────────────────────
        self.offset_filtrado  = 0.0
        self.angulo_filtrado  = 0.0
        self.calidad_filtrada = 0.0
        self.tiene_valor      = False
        self.frames_sin_deteccion = 0          # NUEVO: para reset de estado
        self.MAX_FRAMES_SIN_DETECCION = 30     # ~1.5s a 20fps

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_lanes = self.create_publisher(
            LaneDetection,
            '/perception/lanes',
            10
        )

        self.pub_debug = self.create_publisher(
            Image,
            '/vision/debug_image',
            10
        )

        self.get_logger().info(
            f'Vision node iniciado | debug={self.debug} '
            f'alpha={self.alpha} umbral_calidad={self.umbral_calidad}'
        )

    # ── Filtro exponencial ────────────────────────────────────────────────────
    def _suavizar(self, previo, nuevo):
        return self.alpha * previo + (1.0 - self.alpha) * nuevo

    # ── Callback principal ────────────────────────────────────────────────────
    def image_callback(self, msg: Image):

        # 1) Convertir imagen ROS → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # 2) Procesamiento de líneas
        try:
            frame_resultado, analisis = detectar_y_analizar_lineas(frame)
        except Exception as e:
            self.get_logger().error(f'Error en detectar_y_analizar_lineas: {e}')
            return

        h, w = frame_resultado.shape[:2]

        datos_izq = analisis['Izquierda']
        datos_cen = analisis['Centro']
        datos_der = analisis['Derecha']

        tiene_izq = datos_izq['intermitencia'] > 0
        tiene_cen = datos_cen['intermitencia'] > 0
        tiene_der = datos_der['intermitencia'] > 0

        # 3) Calcular offset y ángulo
        if tiene_izq and tiene_der:
            x_centro_carril = 0.5 * (datos_izq['x_base'] + datos_der['x_base'])
            offset_pix      = x_centro_carril - (w / 2.0)
            offset_norm     = offset_pix / (w / 2.0)
            m               = 0.5 * (datos_izq['pendiente_prom'] + datos_der['pendiente_prom'])
            zona_ref        = 'CentroGeom'
        else:
            zona_ref = 'Centro'
            if datos_cen['intermitencia'] == 0:
                if datos_izq['intermitencia'] > datos_der['intermitencia']:
                    if datos_izq['intermitencia'] > 0:
                        zona_ref = 'Izquierda'
                elif datos_der['intermitencia'] > 0:
                    zona_ref = 'Derecha'

            datos       = analisis[zona_ref] if zona_ref != 'CentroGeom' else datos_cen
            m           = datos['pendiente_prom']
            offset_pix  = datos['x_base'] - (w / 2.0)
            offset_norm = offset_pix / (w / 2.0)

        offset_norm = max(-1.5, min(1.5, offset_norm))
        angulo      = math.atan(m)

        # 4) Calidad de detección
        inter_total = (
            datos_izq['intermitencia'] +
            datos_cen['intermitencia'] +
            datos_der['intermitencia']
        )
        calidad_det = min(1.0, inter_total / 15.0)

        if tiene_izq and tiene_der:
            calidad = max(calidad_det, 0.8)
        elif tiene_cen and (tiene_izq or tiene_der):
            calidad = max(calidad_det * 0.9, 0.6)
        elif tiene_izq or tiene_cen or tiene_der:
            calidad = max(calidad_det * 0.7, 0.3)
        else:
            calidad = 0.0

        # 5) Filtro exponencial con reset automático
        if calidad < self.umbral_calidad:
            self.get_logger().warn(
                f'Calidad baja ({calidad:.2f}) zona={zona_ref} — usando valores filtrados'
            )
            self.frames_sin_deteccion += 1

            if self.tiene_valor:
                self.calidad_filtrada = self._suavizar(self.calidad_filtrada, 0.0)
            else:
                self.offset_filtrado  = 0.0
                self.angulo_filtrado  = 0.0
                self.calidad_filtrada = 0.0

            # Reset del estado filtrado tras N frames consecutivos sin detección
            if self.frames_sin_deteccion >= self.MAX_FRAMES_SIN_DETECCION:
                self.tiene_valor = False
                self.offset_filtrado  = 0.0
                self.angulo_filtrado  = 0.0
                self.calidad_filtrada = 0.0
                self.get_logger().warn(
                    f'Sin detección por {self.frames_sin_deteccion} frames — '
                    f'estado filtrado reseteado'
                )
        else:
            self.frames_sin_deteccion = 0
            if not self.tiene_valor:
                self.offset_filtrado  = offset_norm
                self.angulo_filtrado  = angulo
                self.calidad_filtrada = calidad
                self.tiene_valor      = True
            else:
                self.offset_filtrado  = self._suavizar(self.offset_filtrado,  offset_norm)
                self.angulo_filtrado  = self._suavizar(self.angulo_filtrado,  angulo)
                self.calidad_filtrada = self._suavizar(self.calidad_filtrada, calidad)

        # 6) Publicar LaneDetection — campos explícitos, sin convenciones implícitas
        lane_msg                    = LaneDetection()
        lane_msg.header.stamp       = self.get_clock().now().to_msg()
        lane_msg.header.frame_id    = self.frame_id

        lane_msg.center_offset      = float(self.offset_filtrado)
        lane_msg.heading_angle      = float(self.angulo_filtrado)   # NUEVO campo
        lane_msg.detection_quality  = float(self.calidad_filtrada)  # NUEVO campo

        # left_coeffs / right_coeffs mantienen pendientes por compatibilidad
        # con posibles consumidores futuros, pero ya no codifican calidad/ángulo
        lane_msg.left_coeffs  = [float(datos_izq['pendiente_prom'])]
        lane_msg.right_coeffs = [float(datos_der['pendiente_prom'])]

        lane_msg.left_detected  = tiene_izq
        lane_msg.right_detected = tiene_der

        self.pub_lanes.publish(lane_msg)

        self.get_logger().debug(
            f'zona={zona_ref} | '
            f'offset={self.offset_filtrado:.3f} '
            f'ang={self.angulo_filtrado:.3f} '
            f'calidad={self.calidad_filtrada:.2f}'
        )

        # 7) Imagen debug
        if self.debug:
            self._draw_debug_overlay(frame_resultado, zona_ref)
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(frame_resultado, encoding='bgr8')
                debug_msg.header = lane_msg.header
                self.pub_debug.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Error publicando debug image: {e}')

    def _draw_debug_overlay(self, frame, zona_ref):
        """Dibuja overlay de telemetría sobre el frame de debug."""
        color_calidad = (
            (0, 255, 0) if self.calidad_filtrada >= self.umbral_calidad
            else (0, 0, 255)
        )
        lines = [
            (f'Zona ref: {zona_ref}',                          (255, 255, 255)),
            (f'Offset:   {self.offset_filtrado:.3f}',          (255, 255, 255)),
            (f'Angulo:   {self.angulo_filtrado:.3f} rad',      (255, 255, 255)),
            (f'Calidad:  {self.calidad_filtrada:.2f}',         color_calidad),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                frame, text,
                (10, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA
            )

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()