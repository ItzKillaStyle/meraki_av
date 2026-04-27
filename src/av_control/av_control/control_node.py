import numpy as np

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Point
from av_interfaces.msg import LaneDetection


# ── Prioridades de comando ────────────────────────────────────────────────────
PRIO_LANE     = 1
PRIO_WAYPOINT = 2
PRIO_DODGE    = 3
PRIO_ESTOP    = 4


class PID:
    """PID discreto con anti-windup y límite de salida."""

    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float, out_max: float):
        self.kp      = kp
        self.ki      = ki
        self.kd      = kd
        self.out_min = out_min
        self.out_max = out_max
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self._integral += error * dt
        max_i = self.out_max / (self.ki + 1e-9)
        self._integral = float(np.clip(self._integral, -max_i, max_i))
        derivative     = (error - self._prev_err) / dt
        self._prev_err = error
        output = (self.kp * error
                  + self.ki * self._integral
                  + self.kd * derivative)
        return float(np.clip(output, self.out_min, self.out_max))


class AckermannDifferential:
    """
    Calcula velocidades individuales de las 4 ruedas con geometría Ackermann.

    Layout de motores (coincide con protocolo STM32):
        m1 = Delantera Izquierda  (FL)
        m2 = Delantera Derecha    (FR)
        m3 = Trasera  Izquierda   (RL)
        m4 = Trasera  Derecha     (RR)

    Convención de signo:
        steering_rad > 0  →  giro IZQUIERDA  (rueda izq interna)
        steering_rad < 0  →  giro DERECHA    (rueda der interna)
    """

    def __init__(self, wheelbase: float, track_width: float):
        self.L = wheelbase    # distancia entre ejes [m]
        self.T = track_width  # distancia entre ruedas izq-der [m]

    def compute(self, steering_rad: float,
                speed_norm: float) -> tuple[float, float, float, float]:
        """
        Retorna (v_FL, v_FR, v_RL, v_RR) normalizadas en -1.0..1.0.
        speed_norm es la velocidad base normalizada del vehículo.
        """
        if abs(steering_rad) < 0.01:
            v = float(np.clip(speed_norm, -1.0, 1.0))
            return v, v, v, v

        # Radio de giro respecto al centro del eje trasero
        R = self.L / np.tan(abs(steering_rad))

        # Radios de cada rueda
        r_inner = max(R - self.T / 2.0, 0.01)   # nunca negativo
        r_outer = R + self.T / 2.0

        # Factores relativos normalizados por la rueda más rápida
        f_inner = r_inner / r_outer   # < 1.0
        f_outer = 1.0                 # rueda externa = velocidad base

        base = float(np.clip(speed_norm, -1.0, 1.0))

        if steering_rad > 0:
            # Izquierda interna, derecha externa
            v_FL = base * f_inner
            v_FR = base * f_outer
            v_RL = base * f_inner
            v_RR = base * f_outer
        else:
            # Derecha interna, izquierda externa
            v_FL = base * f_outer
            v_FR = base * f_inner
            v_RL = base * f_outer
            v_RR = base * f_inner

        return (
            float(np.clip(v_FL, -1.0, 1.0)),
            float(np.clip(v_FR, -1.0, 1.0)),
            float(np.clip(v_RL, -1.0, 1.0)),
            float(np.clip(v_RR, -1.0, 1.0)),
        )


class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('wheelbase',          0.65)  
        self.declare_parameter('track_width',        0.55)
        self.declare_parameter('max_speed_ms',       2.5)
        self.declare_parameter('max_steering_rad',   0.5)
        self.declare_parameter('control_hz',         20.0)

        self.declare_parameter('pid_lane_kp',        0.8)
        self.declare_parameter('pid_lane_ki',        0.02)
        self.declare_parameter('pid_lane_kd',        0.15)

        self.declare_parameter('pid_wp_kp',          1.0)
        self.declare_parameter('pid_wp_ki',          0.01)
        self.declare_parameter('pid_wp_kd',          0.20)

        self.declare_parameter('speed_lane_ms',      0.40)
        self.declare_parameter('speed_waypoint_ms',  0.35)
        self.declare_parameter('speed_dodge_ms',     0.20)
        self.declare_parameter('dodge_steer_factor', 0.60)

        self.wheelbase      = self.get_parameter('wheelbase').value
        self.track_width    = self.get_parameter('track_width').value
        self.max_speed      = self.get_parameter('max_speed_ms').value
        self.max_steer      = self.get_parameter('max_steering_rad').value
        self.control_hz     = self.get_parameter('control_hz').value
        self.speed_lane     = self.get_parameter('speed_lane_ms').value
        self.speed_waypoint = self.get_parameter('speed_waypoint_ms').value
        self.speed_dodge    = self.get_parameter('speed_dodge_ms').value
        self.dodge_factor   = self.get_parameter('dodge_steer_factor').value

        # ── Subsistemas ───────────────────────────────────────────────────────
        self.ackermann = AckermannDifferential(self.wheelbase, self.track_width)

        self.pid_lane = PID(
            self.get_parameter('pid_lane_kp').value,
            self.get_parameter('pid_lane_ki').value,
            self.get_parameter('pid_lane_kd').value,
            -self.max_steer, self.max_steer)

        self.pid_wp = PID(
            self.get_parameter('pid_wp_kp').value,
            self.get_parameter('pid_wp_ki').value,
            self.get_parameter('pid_wp_kd').value,
            -self.max_steer, self.max_steer)

        # ── Estado ────────────────────────────────────────────────────────────
        self.emergency     = False
        self.lane_offset   = 0.0
        self.lane_quality  = 0.0
        self.lane_detected = False
        self.dodge_dir     = 0.0
        self.dodge_active  = False
        self.wp_steer      = 0.0
        self.wp_speed      = 0.0
        self.wp_active     = False
        self.current_prio  = PRIO_LANE
        self.last_time     = self.get_clock().now()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Bool, '/emergency_stop', self.cb_estop, 10)
        self.create_subscription(
            LaneDetection, '/perception/lanes', self.cb_lanes, 10)
        self.create_subscription(
            Point, '/perception/dodge_direction', self.cb_dodge, 10)
        self.create_subscription(
            AckermannDriveStamped, '/control/cmd', self.cb_ackermann, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        # 5 valores → [servo_norm, m1_FL, m2_FR, m3_RL, m4_RR]
        self.pub_pwm = self.create_publisher(
            Float32MultiArray, '/control/pwm_cmd', 10)

        self.pub_ackermann_out = self.create_publisher(
            AckermannDriveStamped, '/control/ackermann_applied', 10)

        self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info(
            f'Control node iniciado — {self.control_hz}Hz | '
            f'wheelbase={self.wheelbase}m track={self.track_width}m | '
            f'Diferencial Ackermann ACTIVO | '
            f'pwm_cmd → [servo, FL, FR, RL, RR]'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_estop(self, msg: Bool):
        self.emergency = msg.data
        if self.emergency:
            self.pid_lane.reset()
            self.pid_wp.reset()
            self.get_logger().warn('EMERGENCIA — motores detenidos')
        else:
            self.get_logger().info('Emergencia desactivada')

    def cb_lanes(self, msg: LaneDetection):
        self.lane_offset   = float(msg.center_offset)
        self.lane_quality  = float(msg.left_coeffs[1]) if msg.left_coeffs else 0.0
        self.lane_detected = msg.left_detected or msg.right_detected

    def cb_dodge(self, msg: Point):
        self.dodge_dir    = float(msg.x)
        self.dodge_active = self.dodge_dir != 0.0

    def cb_ackermann(self, msg: AckermannDriveStamped):
        self.wp_steer  = float(msg.drive.steering_angle)
        self.wp_speed  = float(msg.drive.speed)
        self.wp_active = self.wp_speed > 0.0

    # ── Loop de control ───────────────────────────────────────────────────────

    def control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # P4: Emergencia
        if self.emergency:
            self._send(0.0, 0.0)
            return

        # P3: Esquive
        if self.dodge_active:
            steer = float(np.clip(
                self.dodge_dir * self.dodge_factor * self.max_steer,
                -self.max_steer, self.max_steer))
            self._send(steer, self.speed_dodge)
            self.current_prio = PRIO_DODGE
            return

        # P2: Waypoints GPS
        if self.wp_active:
            steer = float(np.clip(self.wp_steer, -self.max_steer, self.max_steer))
            speed = float(np.clip(self.wp_speed, 0.0, self.max_speed))
            if self.lane_detected and self.lane_quality > 0.3:
                correction = self.pid_lane.update(self.lane_offset, dt) * 0.3
                steer = float(np.clip(
                    steer + correction, -self.max_steer, self.max_steer))
            self._send(steer, speed)
            self.current_prio = PRIO_WAYPOINT
            return

        # P1: Seguimiento de carril
        if self.lane_detected and self.lane_quality > 0.3:
            steer = self.pid_lane.update(self.lane_offset, dt)
            self._send(steer, self.speed_lane)
            self.current_prio = PRIO_LANE
            return

        # Sin comando válido
        self._send(0.0, 0.0)

    # ── Publicación con diferencial Ackermann ─────────────────────────────────

    def _send(self, steering_rad: float, speed_ms: float):
        # Convertir radianes a grados 0-180
        # -max_steer → 0°, 0 → 90°, +max_steer → 180°
        servo_deg = 90.0 + float(np.degrees(
            np.clip(steering_rad, -self.max_steer, self.max_steer)
        ))
        servo_deg = float(np.clip(servo_deg, 0.0, 180.0))

        speed_norm = float(np.clip(speed_ms / self.max_speed, -1.0, 1.0))
        v_FL, v_FR, v_RL, v_RR = self.ackermann.compute(steering_rad, speed_norm)

        cmd      = Float32MultiArray()
        cmd.data = [servo_deg, v_FL, v_FR, v_RL, v_RR]
        self.pub_pwm.publish(cmd)

        ack                      = AckermannDriveStamped()
        ack.header.stamp         = self.get_clock().now().to_msg()
        ack.drive.steering_angle = steering_rad
        ack.drive.speed          = speed_ms
        self.pub_ackermann_out.publish(ack)

    def destroy_node(self):
        self._send(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()