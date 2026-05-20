from enum import Enum, auto
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from av_interfaces.msg import LaneDetection, TrafficSign, ObstacleArray


# ── Nombres de señales ────────────────────────────────────────────────────────
class SignName:
    """
    Nombres exactos que publica traffic_sign_node en msg.sign_type.
    Coinciden con las claves de SIGN_CLASSES en traffic_sign_node.py.
    """
    CEDA_EL_PASO       = 'Ceda el paso'
    CRUCE_PEATONAL     = 'Cruce peatonal'
    PARADA_BUS         = 'Parada de bus'
    PARE               = 'Pare'
    PROHIBIDO_GIRO_U   = 'Prohibido el giro en U'
    PROHIBIDO_PARQUEAR = 'Prohibido parquear'
    PROXIMIDAD_CRUCE   = 'Proximidad cruce peatonal'
    SEMAFORO_ROJO      = 'Semaforo peatonal rojo'
    SEMAFORO_VERDE     = 'Semaforo peatonal verde'
    VELOCIDAD_30       = 'Velocidad maxima 30km'

    # Solo loggean, no disparan transición de estado
    LOG_ONLY = {PARADA_BUS, PROHIBIDO_GIRO_U, PROHIBIDO_PARQUEAR, VELOCIDAD_30}


# ── Estados ───────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE               = auto()
    LANE_FOLLOW        = auto()
    DODGE              = auto()
    YIELD              = auto()   # Ceda el paso
    CROSSWALK_WAIT     = auto()   # Cruce peatonal / Proximidad cruce
    STOP_SIGN          = auto()   # Pare
    TRAFFIC_LIGHT_STOP = auto()   # Semáforo rojo
    WAYPOINT_NAV       = auto()
    EMERGENCY_STOP     = auto()


class SignFilter:
    """
    Filtro de ventana de tiempo: una señal se confirma solo si
    se detecta con conf > threshold durante al menos confirm_s segundos
    consecutivos. Se resetea si la señal cambia o la confianza baja.
    """
    def __init__(self, confirm_s: float = 0.5, threshold: float = 0.7):
        self.confirm_s  = confirm_s
        self.threshold  = threshold
        self._candidate = None      # señal candidata actual
        self._since     = None      # timestamp de inicio de ventana
        self._confirmed = None      # última señal confirmada
        self._conf      = 0.0

    def update(self, name: str, conf: float, now_s: float) -> tuple[str | None, float]:
        """
        Actualiza el filtro. Retorna (nombre_confirmado, confianza) si la
        señal se confirma, o (None, 0.0) si aún no hay confirmación.
        """
        if conf >= self.threshold:
            if name != self._candidate:
                # Nueva señal candidata — reinicia ventana
                self._candidate = name
                self._since     = now_s
                self._conf      = conf
            else:
                self._conf = conf
                if now_s - self._since >= self.confirm_s:
                    if name != self._confirmed:
                        self._confirmed = name
                        return name, conf
        else:
            # Confianza bajó — resetea candidato
            self._candidate = None
            self._since     = None
            self._conf      = 0.0

        return None, 0.0

    def reset_confirmed(self):
        self._confirmed = None
        self._candidate = None
        self._since     = None


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('loop_hz',              20.0)
        self.declare_parameter('stop_sign_wait_s',     10.0)
        self.declare_parameter('yield_speed_factor',   0.4)   # fracción de vel normal
        self.declare_parameter('crosswalk_timeout_s',  15.0)  # max espera en cruce
        self.declare_parameter('sign_confirm_s',       0.5)   # ventana filtro
        self.declare_parameter('sign_conf_threshold',  0.7)   # umbral confianza
        self.declare_parameter('dodge_clear_dist',     1.0)
        self.declare_parameter('obstacle_stop_dist',   0.3)
        self.declare_parameter('obstacle_dodge_dist',  0.8)
        self.declare_parameter('waypoint_reach_m',     0.5)
        self.declare_parameter('nominal_speed',        1.5)   # m/s base

        self.stop_sign_wait    = self.get_parameter('stop_sign_wait_s').value
        self.yield_speed_factor= self.get_parameter('yield_speed_factor').value
        self.crosswalk_timeout = self.get_parameter('crosswalk_timeout_s').value
        self.dodge_clear_dist  = self.get_parameter('dodge_clear_dist').value
        self.obs_stop_dist     = self.get_parameter('obstacle_stop_dist').value
        self.obs_dodge_dist    = self.get_parameter('obstacle_dodge_dist').value
        self.waypoint_reach_m  = self.get_parameter('waypoint_reach_m').value
        self.nominal_speed     = self.get_parameter('nominal_speed').value

        # ── Estado ────────────────────────────────────────────────────────────
        self.state          = State.IDLE
        self.prev_state     = State.IDLE
        self.state_entry_t  = self.get_clock().now()

        # ── Datos de percepción ───────────────────────────────────────────────
        self.emergency          = False
        self.obstacle_min_dist  = 999.0
        self.dodge_dir          = 0.0
        self.dodge_clear        = False
        self.lane_detected      = False
        self.waypoint_active    = False
        self.waypoint_dist      = 999.0
        self.wp_steer           = 0.0
        self.wp_speed           = 0.0

        # Señal activa confirmada
        self.active_sign        = None
        self.active_sign_conf   = 0.0

        # ── Filtro de señales ─────────────────────────────────────────────────
        confirm_s   = self.get_parameter('sign_confirm_s').value
        threshold   = self.get_parameter('sign_conf_threshold').value
        self.sign_filter = SignFilter(confirm_s=confirm_s, threshold=threshold)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Bool,            '/emergency_stop',          self.cb_estop,     10)
        self.create_subscription(ObstacleArray,   '/perception/obstacles',    self.cb_obstacles, 10)
        self.create_subscription(Point,           '/perception/dodge_direction', self.cb_dodge,  10)
        self.create_subscription(LaneDetection,   '/perception/lanes',        self.cb_lanes,     10)
        self.create_subscription(TrafficSign,     '/perception/traffic_sign', self.cb_sign,      10)
        self.create_subscription(AckermannDriveStamped, '/planning/waypoint_cmd', self.cb_waypoint, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cmd   = self.create_publisher(AckermannDriveStamped, '/control/cmd',      10)
        self.pub_state = self.create_publisher(String,                '/behavior/state',   10)
        self.pub_estop = self.create_publisher(Bool,                  '/emergency_stop',   10)
        self.pub_sign  = self.create_publisher(String,                '/behavior/sign',    10)  # señal activa

        # ── Timer ─────────────────────────────────────────────────────────────
        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self.behavior_loop)

        self.state = State.LANE_FOLLOW
        self.get_logger().info('Behavior node iniciado → LANE_FOLLOW')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_estop(self, msg: Bool):
        self.emergency = msg.data

    def cb_obstacles(self, msg: ObstacleArray):
        self.obstacle_min_dist = min((o.distance for o in msg.obstacles), default=999.0)

    def cb_dodge(self, msg: Point):
        self.dodge_dir   = float(msg.x)
        self.dodge_clear = (self.dodge_dir == 0.0 and float(msg.y) > self.dodge_clear_dist)

    def cb_lanes(self, msg: LaneDetection):
        self.lane_detected = msg.left_detected or msg.right_detected

    def cb_sign(self, msg: TrafficSign):
        now_s = self.get_clock().now().nanoseconds / 1e9
        name  = str(msg.sign_type) if msg.sign_type and msg.sign_type != 'none' else ''
        conf  = float(msg.confidence)

        if not name:  # NO_SIGN — solo resetea candidato si no hay nada
            self.sign_filter.update('', 0.0, now_s)
            return

        confirmed_name, confirmed_conf = self.sign_filter.update(name, conf, now_s)

        if confirmed_name:
            self.active_sign      = confirmed_name
            self.active_sign_conf = confirmed_conf

            # Solo loggear
            if confirmed_name in SignName.LOG_ONLY:
                self.get_logger().info(f'Señal detectada (solo log): {confirmed_name}')
                pub = String(); pub.data = confirmed_name
                self.pub_sign.publish(pub)
                self.active_sign = None  # no dispara transición
                return

            self.get_logger().info(
                f'Señal confirmada: {confirmed_name} (conf={confirmed_conf:.2f})')
            pub = String(); pub.data = confirmed_name
            self.pub_sign.publish(pub)

    def cb_waypoint(self, msg: AckermannDriveStamped):
        self.wp_steer        = float(msg.drive.steering_angle)
        self.wp_speed        = float(msg.drive.speed)
        self.waypoint_active = self.wp_speed > 0.0
        self.waypoint_dist   = float(msg.drive.steering_angle_velocity) \
            if msg.drive.steering_angle_velocity > 0 else 999.0

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        if new_state != self.state:
            self.get_logger().info(f'Estado: {self.state.name} → {new_state.name}')
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

    def _clear_sign(self):
        self.active_sign      = None
        self.active_sign_conf = 0.0
        self.sign_filter.reset_confirmed()

    def _check_emergency(self) -> bool:
        if self.emergency or self.obstacle_min_dist <= self.obs_stop_dist:
            self._transition(State.EMERGENCY_STOP)
            return True
        return False

    def _check_dodge(self) -> bool:
        if self.obstacle_min_dist <= self.obs_dodge_dist and self.dodge_dir != 0.0:
            self._transition(State.DODGE)
            return True
        return False

    # ── Loop principal ────────────────────────────────────────────────────────

    def behavior_loop(self):
        msg = String(); msg.data = self.state.name
        self.pub_state.publish(msg)

        if self._check_emergency():
            return

        if self.state == State.IDLE:
            self._state_idle()
        elif self.state == State.LANE_FOLLOW:
            self._state_lane_follow()
        elif self.state == State.DODGE:
            self._state_dodge()
        elif self.state == State.YIELD:
            self._state_yield()
        elif self.state == State.CROSSWALK_WAIT:
            self._state_crosswalk_wait()
        elif self.state == State.STOP_SIGN:
            self._state_stop_sign()
        elif self.state == State.TRAFFIC_LIGHT_STOP:
            self._state_traffic_light()
        elif self.state == State.WAYPOINT_NAV:
            self._state_waypoint_nav()
        elif self.state == State.EMERGENCY_STOP:
            self._state_emergency()

    # ── Estados ───────────────────────────────────────────────────────────────

    def _state_idle(self):
        self._send_cmd(0.0, 0.0)

    def _state_lane_follow(self):
        if self._check_dodge():
            return

        sign = self.active_sign

        if sign == SignName.PARE:
            self._transition(State.STOP_SIGN)
            return

        if sign == SignName.SEMAFORO_ROJO:
            self._transition(State.TRAFFIC_LIGHT_STOP)
            return

        if sign in (SignName.CRUCE_PEATONAL, SignName.PROXIMIDAD_CRUCE):
            self._transition(State.CROSSWALK_WAIT)
            return

        if sign == SignName.CEDA_EL_PASO:
            self._transition(State.YIELD)
            return

        if self.waypoint_active:
            self._transition(State.WAYPOINT_NAV)
            return

        # av_control lleva el PID de carril — no enviamos cmd aquí

    def _state_dodge(self):
        if self._check_emergency():
            return
        if self.dodge_clear:
            self.get_logger().info('Esquive completado → LANE_FOLLOW')
            self._transition(State.LANE_FOLLOW)

    def _state_yield(self):
        """
        Ceda el paso: reduce velocidad a yield_speed_factor * nominal_speed.
        Si no hay obstáculo adelante vuelve a LANE_FOLLOW.
        """
        if self._check_emergency():
            return

        if self.obstacle_min_dist > self.obs_dodge_dist:
            # Camino libre — continúa a velocidad reducida y retoma
            self.get_logger().info('Ceda el paso: camino libre → LANE_FOLLOW')
            self._clear_sign()
            self._transition(State.LANE_FOLLOW)
            return

        # Obstáculo presente — avanza lento
        reduced_speed = self.nominal_speed * self.yield_speed_factor
        self._send_cmd(0.0, reduced_speed)

    def _state_crosswalk_wait(self):
        """
        Cruce peatonal / Proximidad cruce: para hasta que no haya obstáculo
        en zona o se supere el timeout.
        """
        self._send_cmd(0.0, 0.0)

        no_obstacle = self.obstacle_min_dist > self.obs_dodge_dist
        timeout     = self._time_in_state() >= self.crosswalk_timeout

        if no_obstacle or timeout:
            if timeout:
                self.get_logger().warn(
                    f'Timeout cruce ({self.crosswalk_timeout}s) → continuando')
            else:
                self.get_logger().info('Cruce libre → LANE_FOLLOW')
            self._clear_sign()
            self._transition(State.LANE_FOLLOW)

    def _state_stop_sign(self):
        """Para stop_sign_wait_s segundos y continúa."""
        self._send_cmd(0.0, 0.0)
        if self._time_in_state() >= self.stop_sign_wait:
            self.get_logger().info(f'PARE completado ({self.stop_sign_wait}s) → LANE_FOLLOW')
            self._clear_sign()
            self._transition(State.LANE_FOLLOW)

    def _state_traffic_light(self):
        """Para hasta ver semáforo verde (timeout 60s)."""
        self._send_cmd(0.0, 0.0)

        if self.active_sign == SignName.SEMAFORO_VERDE:
            self.get_logger().info('Semáforo VERDE → LANE_FOLLOW')
            self._clear_sign()
            self._transition(State.LANE_FOLLOW)
            return

        if self._time_in_state() > 60.0:
            self.get_logger().warn('Timeout semáforo (60s) → continuando')
            self._clear_sign()
            self._transition(State.LANE_FOLLOW)

    def _state_waypoint_nav(self):
        if self._check_emergency():
            return
        if self._check_dodge():
            return
        if self.waypoint_dist <= self.waypoint_reach_m:
            self.get_logger().info('Waypoint alcanzado → LANE_FOLLOW')
            self.waypoint_active = False
            self._transition(State.LANE_FOLLOW)
            return
        if not self.waypoint_active:
            self._transition(State.LANE_FOLLOW)

    def _state_emergency(self):
        self._send_cmd(0.0, 0.0)
        estop = Bool(); estop.data = True
        self.pub_estop.publish(estop)

        if not self.emergency and self.obstacle_min_dist > self.obs_dodge_dist:
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