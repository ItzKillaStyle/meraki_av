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

#!/usr/bin/env python3


class ObstacleNode(Node):

    def __init__(self):
        super().__init__('obstacle_node')

        # ─────────────────────────────────────────────────────────────
        # Parámetros
        # ─────────────────────────────────────────────────────────────

        self.declare_parameter('range_min', 0.3)
        self.declare_parameter('range_max', 8.0)

        self.declare_parameter('stop_dist_front', 1.50)
        self.declare_parameter('stop_dist_lateral', 0.80)

        self.declare_parameter('warn_dist_front', 1.50)
        self.declare_parameter('warn_dist_lateral', 0.80)

        self.declare_parameter('front_half_angle', 45.0)
        self.declare_parameter('rear_half_angle', 45.0)

        self.declare_parameter('cluster_eps', 0.20)
        self.declare_parameter('min_cluster_pts', 4)

        self.declare_parameter('dodge_angle_deg', 30.0)

        # ROTACIÓN DEL LIDAR
        # 180 = lidar montado al revés
        self.declare_parameter('angle_offset_deg', 180.0)

        # invertir sentido angular (CW/CCW)
        self.declare_parameter('invert_direction', False)

        # ─────────────────────────────────────────────────────────────
        # Leer parámetros
        # ─────────────────────────────────────────────────────────────

        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value

        self.stop_dist_front = self.get_parameter('stop_dist_front').value
        self.stop_dist_lateral = self.get_parameter('stop_dist_lateral').value

        self.warn_dist_front = self.get_parameter('warn_dist_front').value
        self.warn_dist_lateral = self.get_parameter('warn_dist_lateral').value

        self.front_half_angle = np.deg2rad(
            self.get_parameter('front_half_angle').value
        )

        self.rear_half_angle = np.deg2rad(
            self.get_parameter('rear_half_angle').value
        )

        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.min_cluster_pts = self.get_parameter('min_cluster_pts').value

        self.dodge_angle = np.deg2rad(
            self.get_parameter('dodge_angle_deg').value
        )

        self.angle_offset = np.deg2rad(
            self.get_parameter('angle_offset_deg').value
        )

        self.invert_direction = self.get_parameter(
            'invert_direction'
        ).value

        # ─────────────────────────────────────────────────────────────
        # Subscribers
        # ─────────────────────────────────────────────────────────────

        self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.cb_scan,
            10
        )

        # ─────────────────────────────────────────────────────────────
        # Publishers
        # ─────────────────────────────────────────────────────────────

        self.pub_obstacles = self.create_publisher(
            ObstacleArray,
            '/perception/obstacles',
            10
        )

        self.pub_estop = self.create_publisher(
            Bool,
            '/emergency_stop',
            10
        )

        self.pub_dodge = self.create_publisher(
            Point,
            '/perception/dodge_direction',
            10
        )

        self.get_logger().info(
            f'Obstacle node iniciado | '
            f'offset={np.degrees(self.angle_offset):.1f}° | '
            f'invert_direction={self.invert_direction}'
        )

    # ─────────────────────────────────────────────────────────────
    # Callback principal
    # ─────────────────────────────────────────────────────────────

    def cb_scan(self, msg: LaserScan):

        ranges = np.array(msg.ranges, dtype=np.float32)

        n = len(ranges)

        # MEJOR QUE LINSPACE
        angles = (
            msg.angle_min +
            np.arange(n) * msg.angle_increment
        )

        # ─────────────────────────────────────────────────────────
        # Correcciones del LIDAR
        # ─────────────────────────────────────────────────────────

        # rotación física
        angles = angles + self.angle_offset

        # invertir CW/CCW
        if self.invert_direction:
            angles = -angles

        # normalizar a [-pi, pi]
        angles = np.arctan2(
            np.sin(angles),
            np.cos(angles)
        )

        # ─────────────────────────────────────────────────────────
        # Filtrar puntos válidos
        # ─────────────────────────────────────────────────────────

        valid = (
            np.isfinite(ranges) &
            (ranges > self.range_min) &
            (ranges < self.range_max)
        )

        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            self._publish_clear(msg.header)
            return

        # debug
        idx = np.argmin(ranges)


        # ─────────────────────────────────────────────────────────
        # Polar → cartesiano
        # ─────────────────────────────────────────────────────────

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        pts = np.column_stack((xs, ys))

        # ─────────────────────────────────────────────────────────
        # Zonas
        # ─────────────────────────────────────────────────────────

        front_mask = (
            np.abs(angles) <= self.front_half_angle
        )

        rear_mask = (
            np.abs(np.abs(angles) - np.pi)
            <= self.rear_half_angle
        )

        lateral_mask = ~front_mask & ~rear_mask

        # ─────────────────────────────────────────────────────────
        # Emergency stop
        # ─────────────────────────────────────────────────────────

        estop = False

        if np.any(front_mask):

            min_front = ranges[front_mask].min()

            if min_front <= self.stop_dist_front:

                estop = True

                self.get_logger().warn(
                    f'OBSTACULO FRONTAL {min_front:.2f}m'
                )

        if np.any(lateral_mask):

            min_lat = ranges[lateral_mask].min()

            if min_lat <= self.stop_dist_lateral:

                estop = True

                self.get_logger().warn(
                    f'OBSTACULO LATERAL {min_lat:.2f}m'
                )

        if np.any(rear_mask):

            min_rear = ranges[rear_mask].min()

            if min_rear <= self.stop_dist_front:

                estop = True

                self.get_logger().warn(
                    f'OBSTACULO TRASERO {min_rear:.2f}m'
                )

        estop_msg = Bool()
        estop_msg.data = estop

        self.pub_estop.publish(estop_msg)

        # ─────────────────────────────────────────────────────────
        # Clustering
        # ─────────────────────────────────────────────────────────

        clusters = self._cluster(pts)

        out_array = ObstacleArray()
        out_array.header = msg.header

        for cluster in clusters:

            cx, cy = cluster.mean(axis=0)

            obs = Obstacle()

            obs.position = Point(
                x=float(cx),
                y=float(cy),
                z=0.0
            )

            obs.distance = float(
                np.sqrt(cx**2 + cy**2)
            )

            obs.width = float(
                cluster[:, 0].max() -
                cluster[:, 0].min()
            )

            obs.height = float(
                cluster[:, 1].max() -
                cluster[:, 1].min()
            )

            obs.obstacle_type = Obstacle.STATIC

            out_array.obstacles.append(obs)

        self.pub_obstacles.publish(out_array)

        # ─────────────────────────────────────────────────────────
        # Esquive
        # ─────────────────────────────────────────────────────────

        self._compute_dodge(
            ranges,
            angles
        )

    # ─────────────────────────────────────────────────────────────
    # Esquive
    # ─────────────────────────────────────────────────────────────

    def _compute_dodge(self, ranges, angles):

        dodge = Point()

        front_mask = (
            np.abs(angles) <= self.front_half_angle
        )

        if not np.any(front_mask):

            dodge.x = 0.0
            dodge.y = self.range_max

            self.pub_dodge.publish(dodge)
            return

        min_front = float(
            ranges[front_mask].min()
        )

        dodge.y = min_front

        if min_front > self.warn_dist_front:

            dodge.x = 0.0

            self.pub_dodge.publish(dodge)
            return

        left_mask = (
            (angles > 0) &
            (angles <= np.pi / 2)
        )

        right_mask = (
            (angles < 0) &
            (angles >= -np.pi / 2)
        )

        mean_left = (
            ranges[left_mask].mean()
            if np.any(left_mask)
            else 0.0
        )

        mean_right = (
            ranges[right_mask].mean()
            if np.any(right_mask)
            else 0.0
        )

        if mean_left > mean_right:

            dodge.x = 1.0

        elif mean_right > mean_left:

            dodge.x = -1.0

        else:

            dodge.x = 0.0

        self.pub_dodge.publish(dodge)

    # ─────────────────────────────────────────────────────────────
    # Clustering
    # ─────────────────────────────────────────────────────────────

    def _cluster(self, pts):

        if len(pts) == 0:
            return []

        clusters = []

        visited = np.zeros(
            len(pts),
            dtype=bool
        )

        for i in range(len(pts)):

            if visited[i]:
                continue

            diffs = pts - pts[i]

            dists = np.einsum(
                'ij,ij->i',
                diffs,
                diffs
            )

            members = np.where(
                dists < self.cluster_eps ** 2
            )[0]

            if len(members) >= self.min_cluster_pts:

                clusters.append(
                    pts[members]
                )

                visited[members] = True

        return clusters

    # ─────────────────────────────────────────────────────────────
    # Limpiar
    # ─────────────────────────────────────────────────────────────

    def _publish_clear(self, header):

        out = ObstacleArray()
        out.header = header

        self.pub_obstacles.publish(out)

        estop_msg = Bool()
        estop_msg.data = False

        self.pub_estop.publish(estop_msg)

        dodge = Point()
        dodge.x = 0.0
        dodge.y = self.range_max
        dodge.z = 0.0

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