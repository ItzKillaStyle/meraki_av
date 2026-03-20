import math
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped
from av_interfaces.srv import SetWaypoints
from std_msgs.msg import Bool, String


# ── Constantes geográficas ────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0


def haversine(lat1: float, lon1: float,
              lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos coordenadas GPS."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing(lat1: float, lon1: float,
            lat2: float, lon2: float) -> float:
    """
    Rumbo en radianes desde punto 1 hacia punto 2.
    0 = Norte, π/2 = Este, -π/2 = Oeste
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon  = lon2 - lon1
    x     = math.sin(dlon) * math.cos(lat2)
    y     = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return math.atan2(x, y)


class PlannerNode(Node):

    def __init__(self):
        super().__init__('planner_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('waypoints_file',    '')
        self.declare_parameter('reach_radius_m',    2.0)
        self.declare_parameter('plan_hz',           5.0)
        self.declare_parameter('max_steering_rad',  0.5)
        self.declare_parameter('nav_speed_ms',      0.35)
        self.declare_parameter('frame_id',          'map')

        self.reach_radius   = self.get_parameter('reach_radius_m').value
        self.plan_hz        = self.get_parameter('plan_hz').value
        self.max_steer      = self.get_parameter('max_steering_rad').value
        self.nav_speed      = self.get_parameter('nav_speed_ms').value
        self.frame_id       = self.get_parameter('frame_id').value

        # ── Estado interno ────────────────────────────────────────────────────
        # Lista de waypoints: [{'lat': float, 'lon': float, 'name': str}, ...]
        self.waypoints: list[dict] = []
        self.current_wp_idx  = 0
        self.active          = False
        self.current_lat     = None
        self.current_lon     = None
        self.current_heading = 0.0   # radianes — del GPS o IMU

        # ── Carga waypoints desde archivo YAML si se especificó ───────────────
        wp_file = self.get_parameter('waypoints_file').value
        if wp_file:
            self._load_waypoints_file(wp_file)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            NavSatFix, '/gps/fix', self.cb_gps, 10)

        # Waypoints dinámicos desde topic — lista de lat,lon separados por coma
        # Formato: "lat1,lon1;lat2,lon2;lat3,lon3"
        self.create_subscription(
            String, '/planning/set_waypoints_str',
            self.cb_waypoints_str, 10)

        # Activar / desactivar navegación
        self.create_subscription(
            Bool, '/planning/active', self.cb_active, 10)

        # ── Servicio para recibir waypoints estructurados ─────────────────────
        self.srv_waypoints = self.create_service(
            SetWaypoints, '/planning/set_waypoints', self.srv_set_waypoints)

        # ── Publishers ────────────────────────────────────────────────────────
        # Comando de navegación hacia av_behavior/av_control
        self.pub_cmd = self.create_publisher(
            AckermannDriveStamped, '/planning/waypoint_cmd', 10)

        # Ruta completa para visualización en RViz
        self.pub_path = self.create_publisher(
            Path, '/planning/path', 10)

        # Waypoint actual para debug
        self.pub_current_wp = self.create_publisher(
            Point, '/planning/current_waypoint', 10)

        # Estado del planner
        self.pub_status = self.create_publisher(
            String, '/planning/status', 10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.plan_hz, self.plan_loop)

        self.get_logger().info(
            f'Planner node iniciado | '
            f'reach_radius={self.reach_radius}m '
            f'nav_speed={self.nav_speed}m/s'
        )

    # ── Carga de waypoints ────────────────────────────────────────────────────

    def _load_waypoints_file(self, path: str):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            wps = data.get('waypoints', [])
            self.waypoints     = wps
            self.current_wp_idx = 0
            self.get_logger().info(
                f'Waypoints cargados desde {path}: {len(wps)} puntos')
            self._publish_path()
        except Exception as e:
            self.get_logger().error(f'Error cargando waypoints: {e}')

    def _parse_waypoints_str(self, s: str) -> list[dict]:
        """Parsea 'lat1,lon1,name1;lat2,lon2;...' → lista de dicts."""
        wps = []
        for i, part in enumerate(s.strip().split(';')):
            fields = part.strip().split(',')
            if len(fields) < 2:
                continue
            try:
                wps.append({
                    'lat':  float(fields[0]),
                    'lon':  float(fields[1]),
                    'name': fields[2].strip() if len(fields) > 2 else f'WP{i}'
                })
            except ValueError:
                continue
        return wps

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_gps(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def cb_active(self, msg: Bool):
        self.active = msg.data
        if self.active:
            self.current_wp_idx = 0
            self.get_logger().info(
                f'Navegación activada — {len(self.waypoints)} waypoints')
        else:
            self.get_logger().info('Navegación desactivada')
            self._send_stop()

    def cb_waypoints_str(self, msg: String):
        wps = self._parse_waypoints_str(msg.data)
        if wps:
            self.waypoints      = wps
            self.current_wp_idx = 0
            self.get_logger().info(
                f'Waypoints recibidos por topic: {len(wps)} puntos')
            self._publish_path()
        else:
            self.get_logger().warn('No se pudieron parsear los waypoints')

    def srv_set_waypoints(self, request, response):
        """Servicio ROS2 para recibir waypoints estructurados."""
        wps = []
        for i, pose in enumerate(request.waypoints.poses):
            wps.append({
                'lat':  pose.position.x,   # lat en x por convención
                'lon':  pose.position.y,   # lon en y
                'name': f'WP{i}'
            })
        if wps:
            self.waypoints      = wps
            self.current_wp_idx = 0
            self._publish_path()
            response.success = True
            response.message = f'{len(wps)} waypoints cargados'
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = 'Lista de waypoints vacía'
        return response

    # ── Loop de planificación ─────────────────────────────────────────────────

    def plan_loop(self):
        # Publica estado
        status      = String()
        status.data = (
            f'active={self.active} '
            f'wp={self.current_wp_idx}/{len(self.waypoints)} '
            f'fix={"yes" if self.current_lat else "no"}'
        )
        self.pub_status.publish(status)

        if not self.active:
            return

        if not self.waypoints:
            self.get_logger().warn('Sin waypoints cargados', throttle_duration_sec=5.0)
            return

        if self.current_lat is None:
            self.get_logger().warn('Sin fix GPS', throttle_duration_sec=5.0)
            return

        if self.current_wp_idx >= len(self.waypoints):
            self.get_logger().info('Ruta completada')
            self.active = False
            self._send_stop()
            return

        # ── Waypoint objetivo actual ──────────────────────────────────────────
        wp = self.waypoints[self.current_wp_idx]
        wp_lat = float(wp['lat'])
        wp_lon = float(wp['lon'])

        dist = haversine(self.current_lat, self.current_lon, wp_lat, wp_lon)

        # ── Publicar waypoint actual ──────────────────────────────────────────
        wp_pt     = Point()
        wp_pt.x   = wp_lat
        wp_pt.y   = wp_lon
        wp_pt.z   = float(dist)
        self.pub_current_wp.publish(wp_pt)

        # ── Detección de llegada ──────────────────────────────────────────────
        if dist <= self.reach_radius:
            self.get_logger().info(
                f'Waypoint {self.current_wp_idx} "{wp.get("name","")}" '
                f'alcanzado ({dist:.1f}m) → siguiente')
            self.current_wp_idx += 1
            return

        # ── Cálculo de steering ───────────────────────────────────────────────
        target_bearing = bearing(
            self.current_lat, self.current_lon, wp_lat, wp_lon)

        # Error angular entre heading actual y bearing al waypoint
        heading_error = target_bearing - self.current_heading
        # Normaliza a -π .. π
        while heading_error >  math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        # Steering proporcional al error de heading
        # Ganancia 0.8 — ajustar en campo
        steer = float(
            max(-self.max_steer,
                min(self.max_steer, 0.8 * heading_error))
        )

        # Velocidad reducida al acercarse al waypoint
        speed = self.nav_speed
        if dist < 2.0:
            speed = self.nav_speed * 0.5

        # ── Publica comando hacia av_behavior ─────────────────────────────────
        cmd                              = AckermannDriveStamped()
        cmd.header.stamp                 = self.get_clock().now().to_msg()
        cmd.drive.steering_angle         = steer
        cmd.drive.speed                  = speed
        # Distancia al waypoint en steering_angle_velocity (campo extra disponible)
        cmd.drive.steering_angle_velocity = dist
        self.pub_cmd.publish(cmd)

        self.get_logger().debug(
            f'WP{self.current_wp_idx} "{wp.get("name","")}" | '
            f'dist={dist:.1f}m bearing={math.degrees(target_bearing):.1f}° '
            f'steer={math.degrees(steer):.1f}° speed={speed:.2f}m/s'
        )

    # ── Publicación de ruta completa ──────────────────────────────────────────

    def _publish_path(self):
        path            = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id

        for wp in self.waypoints:
            pose                  = PoseStamped()
            pose.header           = path.header
            pose.pose.position.x  = float(wp['lat'])
            pose.pose.position.y  = float(wp['lon'])
            pose.pose.position.z  = 0.0
            path.poses.append(pose)

        self.pub_path.publish(path)

    def _send_stop(self):
        cmd       = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.speed  = 0.0
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()