#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, Bool, String
from geometry_msgs.msg import Point
from av_interfaces.msg import LaneDetection


PRIO_TELEOP   = 0
PRIO_LANE     = 1
PRIO_WAYPOINT = 2
PRIO_DODGE    = 3
PRIO_ESTOP    = 4

MODEL_ACKERMANN    = 'ackermann'
MODEL_DIFFERENTIAL = 'differential'


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self._integral = 0.0; self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0; self._prev_err = 0.0

    def update(self, error, dt):
        if dt <= 0.0: return 0.0
        self._integral += error * dt
        max_i = self.out_max / (self.ki + 1e-9)
        self._integral = float(np.clip(self._integral, -max_i, max_i))
        derivative = (error - self._prev_err) / dt
        self._prev_err = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, self.out_min, self.out_max))


class AckermannDifferential:
    """Geometría Ackermann — ruedas internas más lentas que externas."""
    def __init__(self, wheelbase, track_width):
        self.L = wheelbase
        self.T = track_width

    def compute(self, steering_rad, speed_norm):
        if abs(steering_rad) < 0.01:
            v = float(np.clip(speed_norm, -1.0, 1.0))
            return v, v, v, v
        R       = self.L / np.tan(abs(steering_rad))
        r_inner = max(R - self.T / 2.0, 0.01)
        r_outer = R + self.T / 2.0
        f_inner = r_inner / r_outer
        base    = float(np.clip(speed_norm, -1.0, 1.0))
        if steering_rad > 0:
            v_FL, v_FR, v_RL, v_RR = base*f_inner, base, base*f_inner, base
        else:
            v_FL, v_FR, v_RL, v_RR = base, base*f_inner, base, base*f_inner
        return (float(np.clip(v_FL,-1,1)), float(np.clip(v_FR,-1,1)),
                float(np.clip(v_RL,-1,1)), float(np.clip(v_RR,-1,1)))


class DifferentialDrive:
    """
    Cinemática diferencial pura — sin servo.
    El steering_rad se mapea a diferencial de velocidad entre lados.
    angular_gain controla qué tan agresivo es el giro.
    """
    def __init__(self, angular_gain=1.0):
        self.angular_gain = angular_gain

    def compute(self, steering_rad, speed_norm):
        base    = float(np.clip(speed_norm, -1.0, 1.0))
        angular = float(np.clip(steering_rad * self.angular_gain, -1.0, 1.0))
        # v_l = base + angular, v_r = base - angular
        v_l = float(np.clip(base + angular, -1.0, 1.0))
        v_r = float(np.clip(base - angular, -1.0, 1.0))
        # Las 4 ruedas siguen el mismo lado
        return v_l, v_r, v_l, v_r


class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        # ── Parámetros existentes ─────────────────────────────────────────────
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
        self.declare_parameter('lane_angle_ff_gain', 0.4)
        self.declare_parameter('umbral_calidad',     0.3)
        self.declare_parameter('dodge_timeout_s',    0.5)
        self.declare_parameter('waypoint_timeout_s', 1.0)

        # ── Parámetros nuevos: modelo de conducción ───────────────────────────
        # 'ackermann' | 'differential'
        self.declare_parameter('drive_model',        MODEL_ACKERMANN)
        # Ganancia angular para el modelo diferencial
        # (cuánto steering_rad afecta al diferencial de ruedas)
        self.declare_parameter('diff_angular_gain',  1.0)

        self.wheelbase          = self.get_parameter('wheelbase').value
        self.track_width        = self.get_parameter('track_width').value
        self.max_speed          = self.get_parameter('max_speed_ms').value
        self.max_steer          = self.get_parameter('max_steering_rad').value
        self.control_hz         = self.get_parameter('control_hz').value
        self.speed_lane         = self.get_parameter('speed_lane_ms').value
        self.speed_waypoint     = self.get_parameter('speed_waypoint_ms').value
        self.speed_dodge        = self.get_parameter('speed_dodge_ms').value
        self.dodge_factor       = self.get_parameter('dodge_steer_factor').value
        self.lane_angle_ff_gain = self.get_parameter('lane_angle_ff_gain').value
        self.umbral_calidad     = self.get_parameter('umbral_calidad').value
        self.dodge_timeout_s    = self.get_parameter('dodge_timeout_s').value
        self.waypoint_timeout_s = self.get_parameter('waypoint_timeout_s').value
        self.drive_model        = self.get_parameter('drive_model').value
        self.diff_angular_gain  = self.get_parameter('diff_angular_gain').value

        # ── Modelos de cinemática ─────────────────────────────────────────────
        self.ackermann_model = AckermannDifferential(self.wheelbase, self.track_width)
        self.diff_model      = DifferentialDrive(self.diff_angular_gain)

        # ── PIDs ──────────────────────────────────────────────────────────────
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
        self.teleop_active   = False
        self.emergency       = False
        self.lane_offset     = 0.0
        self.lane_angle      = 0.0
        self.lane_quality    = 0.0
        self.lane_detected   = False
        self.dodge_dir       = 0.0
        self.dodge_active    = False
        self.dodge_last_time = None
        self.wp_steer        = 0.0
        self.wp_speed        = 0.0
        self.wp_active       = False
        self.wp_last_time    = None
        self.current_prio    = PRIO_LANE
        self.last_time       = self.get_clock().now()
        self._prev_prio      = PRIO_LANE

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Bool,                '/teleop/active',           self.cb_teleop,   10)
        self.create_subscription(Bool,                '/emergency_stop',          self.cb_estop,    10)
        self.create_subscription(LaneDetection,       '/perception/lanes',        self.cb_lanes,    10)
        self.create_subscription(Point,               '/perception/dodge_direction', self.cb_dodge, 10)
        self.create_subscription(AckermannDriveStamped, '/planning/waypoint_cmd', self.cb_ackermann,10)

        # Cambio de modelo en caliente — publica 'ackermann' o 'differential'
        self.create_subscription(String, '/control/drive_model', self.cb_drive_model, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_pwm          = self.create_publisher(Float32MultiArray,    '/control/pwm_cmd',          10)
        self.pub_ackermann_out= self.create_publisher(AckermannDriveStamped,'/control/ackermann_applied', 10)
        # Publica el modelo activo para que HMI/monitor lo vean
        self.pub_model_active = self.create_publisher(String,               '/control/drive_model_active',10)

        self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info(
            f'Control node | model={self.drive_model} | '
            f'{self.control_hz}Hz | wheelbase={self.wheelbase}m | '
            f'track={self.track_width}m | diff_gain={self.diff_angular_gain}'
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _elapsed_s(self, stamp):
        if stamp is None: return float('inf')
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _on_mode_change(self, new_prio):
        if new_prio != self._prev_prio:
            self.pid_lane.reset()
            self.pid_wp.reset()
            self.get_logger().info(f'Cambio de modo: {self._prev_prio} → {new_prio}')
            self._prev_prio = new_prio

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_teleop(self, msg):
        prev = self.teleop_active
        self.teleop_active = msg.data
        if self.teleop_active and not prev:
            self.pid_lane.reset(); self.pid_wp.reset()
            self.get_logger().info('TELEOP activado')
        elif not self.teleop_active and prev:
            self.get_logger().info('TELEOP desactivado')

    def cb_estop(self, msg):
        self.emergency = msg.data
        if self.emergency:
            self.pid_lane.reset(); self.pid_wp.reset()
            self.get_logger().warn('EMERGENCIA')

    def cb_lanes(self, msg):
        self.lane_offset   = float(msg.center_offset)
        self.lane_angle    = float(msg.heading_angle)
        self.lane_quality  = float(msg.detection_quality)
        self.lane_detected = msg.left_detected or msg.right_detected

    def cb_dodge(self, msg):
        self.dodge_dir       = float(msg.x)
        self.dodge_active    = self.dodge_dir != 0.0
        self.dodge_last_time = self.get_clock().now()

    def cb_ackermann(self, msg):
        self.wp_steer     = float(msg.drive.steering_angle)
        self.wp_speed     = float(msg.drive.speed)
        self.wp_active    = self.wp_speed > 0.0
        self.wp_last_time = self.get_clock().now()

    def cb_drive_model(self, msg):
        """Cambia el modelo de conducción en caliente."""
        requested = msg.data.strip().lower()
        if requested not in (MODEL_ACKERMANN, MODEL_DIFFERENTIAL):
            self.get_logger().warn(
                f'Modelo desconocido: "{requested}". '
                f'Usa "{MODEL_ACKERMANN}" o "{MODEL_DIFFERENTIAL}"'
            )
            return
        if requested == self.drive_model:
            return
        prev = self.drive_model
        self.drive_model = requested
        # Reset PIDs al cambiar para evitar windup cruzado
        self.pid_lane.reset(); self.pid_wp.reset()
        self.get_logger().info(f'Modelo de conducción: {prev} → {self.drive_model}')
        out = String(); out.data = self.drive_model
        self.pub_model_active.publish(out)

    # ── Loop de control ───────────────────────────────────────────────────────

    def control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.dodge_active and self._elapsed_s(self.dodge_last_time) > self.dodge_timeout_s:
            self.dodge_active = False
            self.get_logger().warn('Dodge timeout')

        if self.wp_active and self._elapsed_s(self.wp_last_time) > self.waypoint_timeout_s:
            self.wp_active = False
            self.get_logger().warn('Waypoint timeout')

        if self.teleop_active:
            self._on_mode_change(PRIO_TELEOP)
            self.current_prio = PRIO_TELEOP
            return

        if self.emergency:
            self._on_mode_change(PRIO_ESTOP)
            self._send(0.0, 0.0)
            self.current_prio = PRIO_ESTOP
            return

        if self.dodge_active:
            self._on_mode_change(PRIO_DODGE)
            steer = float(np.clip(
                self.dodge_dir * self.dodge_factor * self.max_steer,
                -self.max_steer, self.max_steer))
            self._send(steer, self.speed_dodge)
            self.current_prio = PRIO_DODGE
            return

        if self.wp_active:
            self._on_mode_change(PRIO_WAYPOINT)
            steer = float(np.clip(self.wp_steer, -self.max_steer, self.max_steer))
            speed = float(np.clip(self.wp_speed, 0.0, self.max_speed))
            if self.lane_detected and self.lane_quality > self.umbral_calidad:
                correction = self.pid_lane.update(self.lane_offset, dt) * 0.3
                steer = float(np.clip(steer + correction, -self.max_steer, self.max_steer))
            self._send(steer, speed)
            self.current_prio = PRIO_WAYPOINT
            return

        if self.lane_detected and self.lane_quality > self.umbral_calidad:
            self._on_mode_change(PRIO_LANE)
            steer_offset = self.pid_lane.update(self.lane_offset, dt)
            steer_angle  = self.lane_angle * self.lane_angle_ff_gain
            steer = float(np.clip(steer_offset + steer_angle, -self.max_steer, self.max_steer))
            self._send(steer, self.speed_lane)
            self.current_prio = PRIO_LANE
            return

        self._send(0.0, 0.0)

    # ── Publicación ───────────────────────────────────────────────────────────

    def _send(self, steering_rad: float, speed_ms: float):
        speed_norm = float(np.clip(speed_ms / self.max_speed, -1.0, 1.0))

        if self.drive_model == MODEL_ACKERMANN:
            servo_deg = 90.0 + float(np.degrees(
                np.clip(steering_rad, -self.max_steer, self.max_steer)))
            servo_deg = float(np.clip(servo_deg, 0.0, 180.0))
            v_FL, v_FR, v_RL, v_RR = self.ackermann_model.compute(steering_rad, speed_norm)
        else:
            # Diferencial puro — servo centrado siempre
            servo_deg = 90.0
            v_FL, v_FR, v_RL, v_RR = self.diff_model.compute(steering_rad, speed_norm)

        cmd      = Float32MultiArray()
        cmd.data = [servo_deg, v_RR, v_RL, v_FL, v_FR]
        self.pub_pwm.publish(cmd)

        ack                      = AckermannDriveStamped()
        ack.header.stamp         = self.get_clock().now().to_msg()
        ack.drive.steering_angle = steering_rad
        ack.drive.speed          = speed_ms
        self.pub_ackermann_out.publish(ack)

        self.get_logger().debug(
            f'[{self.drive_model}] prio={self.current_prio} | '
            f'servo={servo_deg:.1f}° | '
            f'FL={v_FL:+.2f} FR={v_FR:+.2f} RL={v_RL:+.2f} RR={v_RR:+.2f}'
        )

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