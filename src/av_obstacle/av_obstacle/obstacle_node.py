import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from av_interfaces.msg import ObstacleArray, Obstacle
from geometry_msgs.msg import Point


# ── Zonas angulares (grados, referencia: 0° = frente del vehículo) ────────────
#
#          FRENTE
#       -45° ... +45°
#   IZQ  ←       →  DER
#  -135° ... +45° ... +135°
#          TRASERO
#        ±135° ... ±180°
#
# En LaserScan: angle_min=-π, angle_max=+π, 0=frente


class ObstacleNode(Node):

    def __init__(self):
        super().__init__('obstacle_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('range_min',          0.15)
        self.declare_parameter('range_max',          8.0)

        # Distancias de alerta por zona
        self.declare_parameter('stop_dist_front',    0.30)   # freno inmediato
        self.declare_parameter('stop_dist_lateral',  0.20)   # lateral más permisivo
        self.declare_parameter('warn_dist_front',    0.80)   # zona de precaución
        self.declare_parameter('warn_dist_lateral',  0.50)

        # Ángulos de zona frontal (grados)
        self.declare_parameter('front_half_angle',   45.0)   # ±45° = 90° frontal
        self.declare_parameter('rear_half_angle',    45.0)   # ±45° trasero

        # Clustering
        self.declare_parameter('cluster_eps',        0.25)   # metros
        self.declare_parameter('min_cluster_pts',    3)

        # Esquive
        self.declare_parameter('dodge_angle_deg',    30.0)   # ángulo de giro esquive

        self.range_min         = self.get_parameter('range_min').value
        self.range_max         = self.get_parameter('range_max').value
        self.stop_dist_front   = self.get_parameter('stop_dist_front').value
        self.stop_dist_lateral = self.get_parameter('stop_dist_lateral').value
        self.warn_dist_front   = self.get_parameter('warn_dist_front').value
        self.warn_dist_lateral = self.get_parameter('warn_dist_lateral').value
        self.front_half_angle  = np.deg2rad(self.get_parameter('front_half_angle').value)
        self.rear_half_angle   = np.deg2rad(self.get_parameter('rear_half_angle').value)
        self.cluster_eps       = self.get_parameter('cluster_eps').value
        self.min_cluster_pts   = self.get_parameter('min_cluster_pts').value
        self.dodge_angle       = np.deg2rad(self.get_parameter('dodge_angle_deg').value)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/lidar/scan', self.cb_scan, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        # Array de obstáculos detectados con posición y distancia
        self.pub_obstacles = self.create_publisher(
            ObstacleArray, '/perception/obstacles', 10)

        # Freno de emergencia — leído por av_control y av_behavior
        self.pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)

        # Dirección de esquive sugerida: +1.0=izquierda, -1.0=derecha, 0.0=ninguna
        self.pub_dodge = self.create_publisher(
            Point, '/perception/dodge_direction', 10)

        self.get_logger().info(
            f'Obstacle node iniciado — '
            f'stop_front={self.stop_dist_front}m '
            f'stop_lateral={self.stop_dist_lateral}m '
            f'warn_front={self.warn_dist_front}m'
        )

    # ── Callback principal ────────────────────────────────────────────────────

    def cb_scan(self, msg: LaserScan):
        ranges  = np.array(msg.ranges, dtype=np.float32)
        n       = len(ranges)
        angles  = np.linspace(msg.angle_min, msg.angle_max, n)

        # Filtrar rangos válidos
        valid = (
            np.isfinite(ranges) &
            (ranges > self.range_min) &
            (ranges < self.range_max)
        )
        ranges  = ranges[valid]
        angles  = angles[valid]

        if len(ranges) == 0:
            self._publish_clear(msg.header)
            return

        # Convertir a cartesiano
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pts = np.column_stack((xs, ys))

        # Clasificar puntos por zona
        front_mask   = np.abs(angles) <= self.front_half_angle
        rear_mask    = np.abs(np.abs(angles) - np.pi) <= self.rear_half_angle
        lateral_mask = ~front_mask & ~rear_mask

        # ── Detección de emergencia por zona ──────────────────────────────────
        estop = False

        if np.any(front_mask):
            min_front = ranges[front_mask].min()
            if min_front <= self.stop_dist_front:
                estop = True
                self.get_logger().warn(
                    f'OBSTACULO FRONTAL a {min_front:.2f}m — FRENO')

        if np.any(lateral_mask):
            min_lat = ranges[lateral_mask].min()
            if min_lat <= self.stop_dist_lateral:
                estop = True
                self.get_logger().warn(
                    f'OBSTACULO LATERAL a {min_lat:.2f}m — FRENO')

        estop_msg      = Bool()
        estop_msg.data = estop
        self.pub_estop.publish(estop_msg)

        # ── Clustering simple ─────────────────────────────────────────────────
        clusters = self._cluster(pts)

        # ── Publicar ObstacleArray ─────────────────────────────────────────────
        out_array        = ObstacleArray()
        out_array.header = msg.header

        for cluster in clusters:
            cx, cy   = cluster.mean(axis=0)
            dist     = float(np.sqrt(cx**2 + cy**2))
            angle_cl = float(np.arctan2(cy, cx))

            obs              = Obstacle()
            obs.position     = Point(x=float(cx), y=float(cy), z=0.0)
            obs.distance     = dist
            obs.width        = float(cluster[:, 0].max() - cluster[:, 0].min())
            obs.height       = float(cluster[:, 1].max() - cluster[:, 1].min())
            obs.obstacle_type = Obstacle.STATIC
            out_array.obstacles.append(obs)

        self.pub_obstacles.publish(out_array)

        # ── Lógica de esquive ─────────────────────────────────────────────────
        self._compute_dodge(ranges, angles, msg.header)

        self.get_logger().debug(
            f'{len(clusters)} obstáculos | estop={estop}'
        )

    # ── Lógica de esquive ─────────────────────────────────────────────────────

    def _compute_dodge(self, ranges, angles, header):
        """
        Determina hacia dónde esquivar cuando hay obstáculo frontal en zona
        de precaución (warn_dist_front).

        Estrategia: compara espacio libre promedio a izquierda vs derecha
        dentro de ±90°. El lado con más espacio libre es el lado de esquive.

        Publica Point:
            x = dirección  (+1=izq, -1=der, 0=libre)
            y = distancia mínima frontal
            z = 0
        """
        dodge     = Point()
        dodge.z   = 0.0

        front_mask = np.abs(angles) <= self.front_half_angle
        if not np.any(front_mask):
            dodge.x = 0.0
            dodge.y = self.range_max
            self.pub_dodge.publish(dodge)
            return

        min_front = float(ranges[front_mask].min())
        dodge.y   = min_front

        if min_front > self.warn_dist_front:
            # Sin obstáculo en zona de precaución — no esquivar
            dodge.x = 0.0
            self.pub_dodge.publish(dodge)
            return

        # Compara espacio libre izquierda vs derecha (0° a ±90°)
        left_mask  = (angles > 0) & (angles <= np.pi / 2)
        right_mask = (angles < 0) & (angles >= -np.pi / 2)

        mean_left  = ranges[left_mask].mean()  if np.any(left_mask)  else 0.0
        mean_right = ranges[right_mask].mean() if np.any(right_mask) else 0.0

        if mean_left > mean_right:
            dodge.x = 1.0   # esquivar izquierda
            self.get_logger().info(
                f'Esquive → IZQUIERDA | libre_izq={mean_left:.2f}m libre_der={mean_right:.2f}m')
        elif mean_right > mean_left:
            dodge.x = -1.0  # esquivar derecha
            self.get_logger().info(
                f'Esquive → DERECHA | libre_izq={mean_left:.2f}m libre_der={mean_right:.2f}m')
        else:
            dodge.x = 0.0   # empate — sin decisión

        self.pub_dodge.publish(dodge)

    # ── Clustering euclidiano simple ──────────────────────────────────────────

    def _cluster(self, pts: np.ndarray) -> list:
        """
        Clustering por distancia euclidiana mínima.
        Retorna lista de arrays numpy, uno por cluster.
        """
        if len(pts) == 0:
            return []

        clusters = []
        visited  = set()

        for i in range(len(pts)):
            if i in visited:
                continue
            dists   = np.linalg.norm(pts - pts[i], axis=1)
            members = np.where(dists < self.cluster_eps)[0]
            if len(members) >= self.min_cluster_pts:
                clusters.append(pts[members])
                visited.update(members.tolist())

        return clusters

    def _publish_clear(self, header):
        """Publica estado limpio cuando no hay puntos válidos."""
        out        = ObstacleArray()
        out.header = header
        self.pub_obstacles.publish(out)

        estop_msg      = Bool()
        estop_msg.data = False
        self.pub_estop.publish(estop_msg)

        dodge      = Point()
        dodge.x    = 0.0
        dodge.y    = self.range_max
        dodge.z    = 0.0
        self.pub_dodge.publish(dodge)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()