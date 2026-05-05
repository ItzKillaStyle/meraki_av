#!/usr/bin/env python3
"""
HC-12 Bridge Node
Lee comandos JSON del HC-12 → publica en ROS
Subscribe sensores + GPS + Path → envía telemetría por HC-12
+ Publisher /control/drive_model para cambiar modelo desde HMI
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String, Bool
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
import serial
import json
import threading
import numpy as np
import time


class HC12Bridge(Node):
    def __init__(self):
        super().__init__('hc12_bridge')

        self.declare_parameter('port', '/dev/ttyHC12')
        self.declare_parameter('baud', 19200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # Publishers → control
        self.pub_teleop     = self.create_publisher(Bool,             '/teleop/active',            10)
        self.pub_pwm        = self.create_publisher(Float32MultiArray,'/control/pwm_cmd',           10)
        self.pub_waypoints  = self.create_publisher(String,           '/planning/set_waypoints_str',10)
        self.pub_active     = self.create_publisher(Bool,             '/planning/active',           10)
        self.pub_drive_model= self.create_publisher(String,           '/control/drive_model',       10)  # NUEVO

        self._teleop_timer    = self.create_timer(0.5, self._teleop_watchdog)
        self._last_cmd_time   = 0.0
        self._current_model   = 'ackermann'  # estado local para log

        # Subscribers → telemetría
        self.create_subscription(Float32, '/ultrasonic/front',
            lambda m: self._cache_tel('uf', m.data), 10)
        self.create_subscription(Float32, '/ultrasonic/rear',
            lambda m: self._cache_tel('ur', m.data), 10)
        self.create_subscription(String, '/imu/front',
            lambda m: self._cache_tel('imu_f', m.data), 10)
        self.create_subscription(String, '/imu/rear',
            lambda m: self._cache_tel('imu_r', m.data), 10)
        self.create_subscription(NavSatFix, '/gps/fix', self.cb_gps, 10)
        self.create_subscription(Path, '/planning/path', self.cb_path, 10)

        # Suscribirse al modelo activo para mantener estado local sincronizado
        self.create_subscription(String, '/control/drive_model_active',
            lambda m: setattr(self, '_current_model', m.data), 10)

        self.tel = {
            'uf': 0.0, 'ur': 0.0,
            'imu_f': {}, 'imu_r': {},
            'gps': {'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
            'path': []
        }

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'HC-12 conectado: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(f'Error HC-12: {e}')
            self.ser = None

        self.create_timer(0.2, self._send_telemetry)
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()

    # ── GPS / Path ────────────────────────────────────────────────────────────

    def cb_gps(self, msg: NavSatFix):
        self.tel['gps'] = {
            'lat': float(msg.latitude),
            'lon': float(msg.longitude),
            'alt': float(msg.altitude),
        }

    def cb_path(self, msg: Path):
        self.tel['path'] = [
            {'lat': float(p.pose.position.x),
             'lon': float(p.pose.position.y)}
            for p in msg.poses
        ]

    # ── Caché telemetría ──────────────────────────────────────────────────────

    def _cache_tel(self, key, val):
        if key in ('imu_f', 'imu_r'):
            try:    self.tel[key] = json.loads(val)
            except: self.tel[key] = {}
        else:
            self.tel[key] = round(float(val), 2)

    # ── TX telemetría ─────────────────────────────────────────────────────────

    def _send_telemetry(self):
        if not self.ser or not self.ser.is_open:
            return
        try:
            imu_f = self.tel['imu_f']
            imu_r = self.tel['imu_r']
            if isinstance(imu_f, str):
                try:    imu_f = json.loads(imu_f)
                except: imu_f = {}
            if isinstance(imu_r, str):
                try:    imu_r = json.loads(imu_r)
                except: imu_r = {}
            payload = json.dumps({
                't':       'tel',
                'uf':      self.tel['uf'],
                'ur':      self.tel['ur'],
                'imu_f':   imu_f,
                'imu_r':   imu_r,
                'gps':     self.tel['gps'],
                'path':    self.tel['path'],
                'model':   self._current_model,   # informativo al HMI
            }) + '\n'
            self.ser.write(payload.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f'Error TX: {e}')

    # ── RX serial con parser balanceado ──────────────────────────────────────

    def _reader(self):
        buf = ''
        while self._running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1); continue
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    buf += self.ser.read(waiting).decode('utf-8', errors='ignore')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        # Extraer todos los JSONs balanceados de la línea
                        i = 0
                        while i < len(line):
                            start = line.find('{', i)
                            if start == -1: break
                            depth = 0
                            for j, c in enumerate(line[start:], start):
                                if c == '{':   depth += 1
                                elif c == '}':
                                    depth -= 1
                                    if depth == 0:
                                        try:
                                            self._parse_cmd(line[start:j+1])
                                        except Exception:
                                            pass
                                        i = j + 1
                                        break
                            else:
                                break
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error RX: {e}')

    # ── Parser de comandos ────────────────────────────────────────────────────

    def _parse_cmd(self, line: str):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            self.get_logger().warning(f'JSON inválido: {line}')
            return

        t = obj.get('t')

        if t == 'cmd':
            s_norm = float(obj.get('s', 0.0))
            s_deg  = float(np.clip(90.0 + s_norm * 90.0, 0.0, 180.0))
            msg = Float32MultiArray()
            msg.data = [
                s_deg,
                float(obj.get('rl', 0.0)),
                float(obj.get('rr', 0.0)),
                float(obj.get('fl', 0.0)),
                float(obj.get('fr', 0.0)),
            ]
            self.pub_pwm.publish(msg)
            teleop_msg = Bool(); teleop_msg.data = True
            self.pub_teleop.publish(teleop_msg)
            self._last_cmd_time = time.time()

        elif t == 'waypoints':
            wps = obj.get('waypoints', [])
            if wps:
                csv = ';'.join(
                    f"{wp['lat']},{wp['lon']},WP{i}"
                    for i, wp in enumerate(wps)
                )
                msg = String(); msg.data = csv
                self.pub_waypoints.publish(msg)
                self.get_logger().info(f'Waypoints → planner: {len(wps)} pts')

        elif t == 'activate':
            msg = Bool(); msg.data = True
            self.pub_active.publish(msg)
            self.get_logger().info('Navegación activada')

        elif t == 'deactivate':
            msg = Bool(); msg.data = False
            self.pub_active.publish(msg)
            self.get_logger().info('Navegación desactivada')

        elif t == 'estop':
            stop = Float32MultiArray()
            stop.data = [90.0, 0.0, 0.0, 0.0, 0.0]
            self.pub_pwm.publish(stop)
            msg = Bool(); msg.data = False
            self.pub_active.publish(msg)
            self.get_logger().warn('E-STOP recibido')

        elif t == 'drive_model':
            # ── NUEVO ────────────────────────────────────────────────────────
            model = obj.get('model', '').strip().lower()
            if model in ('ackermann', 'differential'):
                msg = String(); msg.data = model
                self.pub_drive_model.publish(msg)
                self._current_model = model
                self.get_logger().info(f'Modelo → {model}')
            else:
                self.get_logger().warning(f'Modelo desconocido: {model}')

    def _teleop_watchdog(self):
        if time.time() - self._last_cmd_time > 1.0:
            msg = Bool(); msg.data = False
            self.pub_teleop.publish(msg)

    def destroy_node(self):
        self._running = False
        if self.ser: self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HC12Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()