from enum import Enum, auto
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from av_interfaces.msg import LaneDetection, TrafficSign, ObstacleArray


# ── Estados ───────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE               = auto()
    LANE_FOLLOW        = auto()
    DODGE              = auto()
    STOP_SIGN          = auto()
    TRAFFIC_LIGHT_STOP = auto()
    WAYPOINT_NAV       = auto()
    EMERGENCY_STOP     = auto()


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('loop_hz',              20.0)
        self.declare_parameter('stop_sign_wait_s',     3.0)
        self.declare_parameter('dodge_clear_dist',     1.0)
        self.declare_parameter('obstacle_stop_dist',   0.3)
        self.declare_parameter('obstacle_dodge_dist',  0.8)
        self.declare_parameter('lane_quality_min',     0.3)
        self.declare_parameter('waypoint_reach_m',     0.5)

        self.stop_sign_wait    = self.get_parameter('stop_sign_wait_s').value
        self.dodge_clear_dist  = self.get_parameter('dodge_clear_dist').value
        self.obs_stop_dist     = self.get_parameter('obstacle_stop_dist').value
        self.obs_dodge_dist    = self.get_parameter('obstacle_dodge_dist').value
        self.lane_quality_min  = self.get_parameter('lane_quality_min').value
        self.waypoint_reach_m  = self.get_parameter('waypoint_reach_m').value

        # ── Estado inicial ────────────────────────────────────────────────────
        self.state          = State.IDLE
        self.prev_state     = State.IDLE
        self.state_entry_t  = self.get_clock().now()

        # ── Datos de sensores / percepción ────────────────────────────────────
        self.emergency          = False
        self.obstacle_min_dist  = 999.0
        self.dodge_dir          = 0.0
        self.dodge_clear        = False
        self.lane_detected      = False
        self.lane_quality       = 0.0
        self.traffic_sign       = TrafficSign.NO_SIGN
        self.traffic_sign_conf  = 0.0
        self.waypoint_active    = False
        self.waypoint_dist      = 999.0
        self.wp_steer           = 0.0
        self.wp_speed           = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Bool,            '/emergency_stop',
            self.cb_estop,   10)
        self.create_subscription(
            ObstacleArray,   '/perception/obstacles',
            self.cb_obstacles, 10)
        self.create_subscription(
            Point,           '/perception/dodge_direction',
            self.cb_dodge,   10)
        self.create_subscription(
            LaneDetection,   '/perception/lanes',
            self.cb_lanes,   10)
        self.create_subscription(
            TrafficSign,     '/perception/traffic_sign',
            self.cb_sign,    10)
        self.create_subscription(
            AckermannDriveStamped, '/planning/waypoint_cmd',
            self.cb_waypoint, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        # Comando Ackermann hacia av_control
        self.pub_cmd = self.create_publisher(
            AckermannDriveStamped, '/control/cmd', 10)

        # Estado actual para debug/monitoring
        self.pub_state = self.create_publisher(
            String, '/behavior/state', 10)

        # Parada de emergencia
        self.pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)

        # ── Timer principal ───────────────────────────────────────────────────
        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self.behavior_loop)

        # Activa automáticamente
        self.state = State.LANE_FOLLOW
        self.get_logger().info('Behavior node iniciado → LANE_FOLLOW')

    # ── Callbacks de percepción ───────────────────────────────────────────────

    def cb_estop(self, msg: Bool):
        self.emergency = msg.data

    def cb_obstacles(self, msg: ObstacleArray):
        if not msg.obstacles:
            self.obstacle_min_dist = 999.0
            return
        self.obstacle_min_dist = min(o.distance for o in msg.obstacles)

    def cb_dodge(self, msg: Point):
        self.dodge_dir   = float(msg.x)
        self.dodge_clear = (self.dodge_dir == 0.0 and
                            float(msg.y) > self.dodge_clear_dist)

    def cb_lanes(self, msg: LaneDetection):
        self.lane_detected = msg.left_detected or msg.right_detected
        self.lane_quality  = float(msg.left_coeffs[1]) if msg.left_coeffs else 0.0

    def cb_sign(self, msg: TrafficSign):
        self.traffic_sign      = msg.id
        self.traffic_sign_conf = float(msg.confidence)

    def cb_waypoint(self, msg: AckermannDriveStamped):
        self.wp_steer       = float(msg.drive.steering_angle)
        self.wp_speed       = float(msg.drive.speed)
        self.waypoint_active = self.wp_speed > 0.0
        # Distancia al waypoint viene en steering_angle_velocity por convención
        self.waypoint_dist  = float(msg.drive.steering_angle_velocity) \
            if msg.drive.steering_angle_velocity > 0 else 999.0

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        if new_state != self.state:
            self.get_logger().info(
                f'Estado: {self.state.name} → {new_state.name}')
            self.prev_state    = self.state
            self.state         = new_state
            self.state_entry_t = self.get_clock().now()

    def _time_in_state(self) -> float:
        return (self.get_clock().now() - self.state_entry_t).nanoseconds / 1e9

    def _send_cmd(self, steering: float = 0.0, speed: float = 0.0):
        cmd                      = AckermannDriveStamped()
        cmd.header.stamp         = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = steering
        cmd.drive.speed          = speed
        self.pub_cmd.publish(cmd)

    def _publish_state(self):
        msg      = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)

    # ── Loop principal de comportamiento ──────────────────────────────────────

    def behavior_loop(self):
        self._publish_state()

        # ── Emergencia siempre tiene prioridad ────────────────────────────────
        if self.emergency or self.obstacle_min_dist <= self.obs_stop_dist:
            self._transition(State.EMERGENCY_STOP)

        # ── Máquina de estados ────────────────────────────────────────────────
        if self.state == State.IDLE:
            self._state_idle()
        elif self.state == State.LANE_FOLLOW:
            self._state_lane_follow()
        elif self.state == State.DODGE:
            self._state_dodge()
        elif self.state == State.STOP_SIGN:
            self._state_stop_sign()
        elif self.state == State.TRAFFIC_LIGHT_STOP:
            self._state_traffic_light()
        elif self.state == State.WAYPOINT_NAV:
            self._state_waypoint_nav()
        elif self.state == State.EMERGENCY_STOP:
            self._state_emergency()

    # ── Implementación de estados ─────────────────────────────────────────────

    def _state_idle(self):
        """Espera activación — no mueve motores."""
        self._send_cmd(0.0, 0.0)

    def _state_lane_follow(self):
        """
        Estado base — av_control maneja el PID de carril.
        Behavior solo evalúa transiciones.
        """
        # Transición → EMERGENCY (obstáculo muy cerca)
        if self.obstacle_min_dist <= self.obs_stop_dist:
            self._transition(State.EMERGENCY_STOP)
            return

        # Transición → DODGE (obstáculo en zona de precaución)
        if self.obstacle_min_dist <= self.obs_dodge_dist and self.dodge_dir != 0.0:
            self._transition(State.DODGE)
            return

        # Transición → STOP_SIGN
        if (self.traffic_sign == TrafficSign.STOP and
                self.traffic_sign_conf > 0.7):
            self._transition(State.STOP_SIGN)
            return

        # Transición → TRAFFIC_LIGHT_STOP
        if (self.traffic_sign == TrafficSign.TRAFFIC_LIGHT_RED and
                self.traffic_sign_conf > 0.7):
            self._transition(State.TRAFFIC_LIGHT_STOP)
            return

        # Transición → WAYPOINT_NAV
        if self.waypoint_active:
            self._transition(State.WAYPOINT_NAV)
            return

        # Estado base activo — av_control hace el trabajo
        # No enviamos cmd aquí para no interferir con el PID de av_control

    def _state_dodge(self):
        """Esquiva obstáculo lateral y retoma carril cuando hay espacio."""
        # Emergencia durante esquive
        if self.obstacle_min_dist <= self.obs_stop_dist:
            self._transition(State.EMERGENCY_STOP)
            return

        # Retorna a carril si el camino está libre
        if self.dodge_clear:
            self.get_logger().info('Esquive completado — retomando carril')
            self._transition(State.LANE_FOLLOW)
            return

        # Publica dirección de esquive hacia av_control
        # av_control interpreta dodge_direction via /perception/dodge_direction
        # que ya publica av_obstacle — solo supervisamos aquí

    def _state_stop_sign(self):
        """Para el vehículo por stop_sign_wait segundos y continúa."""
        self._send_cmd(0.0, 0.0)

        if self._time_in_state() >= self.stop_sign_wait:
            self.get_logger().info(
                f'STOP completado ({self.stop_sign_wait}s) → retomando')
            # Limpia la señal para no re-detectar
            self.traffic_sign      = TrafficSign.NO_SIGN
            self.traffic_sign_conf = 0.0
            self._transition(State.LANE_FOLLOW)

    def _state_traffic_light(self):
        """Espera mientras el semáforo esté en rojo."""
        self._send_cmd(0.0, 0.0)

        # Semáforo cambió a verde
        if self.traffic_sign == TrafficSign.TRAFFIC_LIGHT_GREEN:
            self.get_logger().info('Semáforo VERDE → retomando')
            self.traffic_sign = TrafficSign.NO_SIGN
            self._transition(State.LANE_FOLLOW)

        # Timeout de seguridad: si pasan 60s sin verde, continúa
        if self._time_in_state() > 60.0:
            self.get_logger().warn('Timeout semáforo (60s) → continuando')
            self._transition(State.LANE_FOLLOW)

    def _state_waypoint_nav(self):
        """Navega hacia waypoint GPS — av_control recibe cmd de av_planner."""
        # Emergencia durante navegación
        if self.obstacle_min_dist <= self.obs_stop_dist:
            self._transition(State.EMERGENCY_STOP)
            return

        # Obstáculo en ruta — esquiva
        if self.obstacle_min_dist <= self.obs_dodge_dist and self.dodge_dir != 0.0:
            self._transition(State.DODGE)
            return

        # Waypoint alcanzado
        if self.waypoint_dist <= self.waypoint_reach_m:
            self.get_logger().info('Waypoint alcanzado → LANE_FOLLOW')
            self.waypoint_active = False
            self._transition(State.LANE_FOLLOW)
            return

        # Waypoint cancelado externamente
        if not self.waypoint_active:
            self._transition(State.LANE_FOLLOW)

    def _state_emergency(self):
        """Freno total — motores a cero hasta que se libere la emergencia."""
        self._send_cmd(0.0, 0.0)

        # Publica emergency_stop para que av_control también frene
        estop      = Bool()
        estop.data = True
        self.pub_estop.publish(estop)

        # Sale de emergencia cuando el obstáculo se aleja y no hay flag externo
        if (not self.emergency and
                self.obstacle_min_dist > self.obs_dodge_dist):
            self.get_logger().info('Emergencia resuelta → LANE_FOLLOW')
            estop.data = False
            self.pub_estop.publish(estop)
            self._transition(State.LANE_FOLLOW)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()