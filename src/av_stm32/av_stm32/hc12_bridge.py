#!/usr/bin/env python3
"""
HC-12 Bridge Node (MEJORADO)
Lee comandos JSON del HC-12 → publica en /control/pwm_cmd
Subscribe sensores (ultrasonidos, IMU) + GPS + Path → envía telemetría por HC-12
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
import serial
import json
import threading
import numpy as np
from std_msgs.msg import Bool
import time

class HC12Bridge(Node):
    def __init__(self):
        super().__init__('hc12_bridge')

        # Parámetros
        self.declare_parameter('port', '/dev/ttyHC12')
        self.declare_parameter('baud', 19200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        #PUblisher - Modo teleoperado
        self.pub_teleop = self.create_publisher(Bool, '/teleop/active', 10)
        self._teleop_timer = self.create_timer(0.5, self._teleop_watchdog)
        self._last_cmd_time = 0.0

        # Publisher → STM32
        self.pub_pwm = self.create_publisher(
            Float32MultiArray, '/control/pwm_cmd', 10)

        # Publishers → Planner (desde HMI)
        self.pub_waypoints = self.create_publisher(
            String, '/planning/set_waypoints_str', 10)
        self.pub_active = self.create_publisher(
            Bool, '/planning/active', 10)

        # Subscribers → telemetría
        self.create_subscription(Float32, '/ultrasonic/front',
            lambda m: self._cache_tel('uf', m.data), 10)
        self.create_subscription(Float32, '/ultrasonic/rear',
            lambda m: self._cache_tel('ur', m.data), 10)

        self.create_subscription(String, '/imu/front',
            lambda m: self._cache_tel('imu_f', m.data), 10)
        self.create_subscription(String, '/imu/rear',
            lambda m: self._cache_tel('imu_r', m.data), 10)

        # ════════════════════════════════════════════════════════════
        # NUEVO: GPS y Path del planner
        # ════════════════════════════════════════════════════════════
        self.create_subscription(NavSatFix, '/gps/fix',
            self.cb_gps, 10)
        self.create_subscription(Path, '/planning/path',
            self.cb_path, 10)

        self.tel = {
            'uf':    0.0,
            'ur':    0.0,
            'imu_f': {},
            'imu_r': {},
            'gps': {'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
            'path': []
        }

        # Serial HC-12
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'HC-12 conectado: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(f'Error HC-12: {e}')
            self.ser = None

        # Timer para enviar telemetría cada 200ms
        self.create_timer(0.2, self._send_telemetry)

        # Thread lectura serial
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()

    # ════════════════════════════════════════════════════════════
    # CALLBACKS NUEVOS: GPS y Path
    # ════════════════════════════════════════════════════════════

    def cb_gps(self, msg: NavSatFix):
        """Recibe GPS del nodo GPS_node y lo cachea."""
        self.tel['gps'] = {
            'lat': float(msg.latitude),
            'lon': float(msg.longitude),
            'alt': float(msg.altitude),
            'sats': 0  # No viene en NavSatFix, pero lo agregamos si lo necesitas
        }
        self.get_logger().debug(
            f'GPS: {msg.latitude:.6f}, {msg.longitude:.6f}, alt={msg.altitude:.1f}m'
        )

    def cb_path(self, msg: Path):
        """Recibe ruta del planner y la cachea."""
        path_list = []
        for pose in msg.poses:
            path_list.append({
                'lat': float(pose.pose.position.x),
                'lon': float(pose.pose.position.y)
            })
        self.tel['path'] = path_list
        self.get_logger().debug(f'Path recibido: {len(path_list)} waypoints')

    # ════════════════════════════════════════════════════════════

    def _cache_tel(self, key, val):
        if key in ('imu_f', 'imu_r'):
            try:
                self.tel[key] = json.loads(val)
            except:
                self.tel[key] = {}
        else:
            self.tel[key] = round(float(val), 2)

    def _send_telemetry(self):
        if not self.ser or not self.ser.is_open:
            return
        try:
            # Parsear imu si viene como string
            imu_f = self.tel['imu_f']
            imu_r = self.tel['imu_r']
            
            if isinstance(imu_f, str):
                try:
                    imu_f = json.loads(imu_f)
                except:
                    imu_f = {}
                    
            if isinstance(imu_r, str):
                try:
                    imu_r = json.loads(imu_r)
                except:
                    imu_r = {}

            # ════════════════════════════════════════════════════════════
            # TELEMETRÍA COMPLETA: sensores + GPS + Path
            # ════════════════════════════════════════════════════════════
            msg = json.dumps({
                't':     'tel',
                'uf':    self.tel['uf'],
                'ur':    self.tel['ur'],
                'imu_f': imu_f,
                'imu_r': imu_r,
                'gps':   self.tel['gps'],      # ← NUEVO
                'path':  self.tel['path']       # ← NUEVO
            }) + '\n'
            self.ser.write(msg.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f'Error TX: {e}')

    def _reader(self):
        import time
        buf = ''
        while self._running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1); continue
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    data = self.ser.read(waiting).decode('utf-8', errors='ignore')
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        # Descartar si hay dos '{' → dos JSONs pegados
                        if line.count('{') > 1:
                            # Quedarse solo con el último JSON completo
                            idx = line.rfind('{')
                            line = line[idx:]
                        if line.startswith('{') and line.endswith('}'):
                            self._parse_cmd(line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error RX: {e}')

    def _parse_cmd(self, line):
        try:
            obj = json.loads(line)
            t = obj.get('t')

            # Control manual (joystick + servo)
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

                teleop_msg = Bool()
                teleop_msg.data = True
                self.pub_teleop.publish(teleop_msg)
                self._last_cmd_time = time.time()

            # Waypoints desde HMI → Planner
            elif t == 'waypoints':
                wps = obj.get('waypoints', [])
                if wps:
                    csv = ';'.join(
                        f"{wp['lat']},{wp['lon']},WP{i}"
                        for i, wp in enumerate(wps)
                    )
                    msg = String()
                    msg.data = csv
                    self.pub_waypoints.publish(msg)
                    self.get_logger().info(f'Waypoints → planner: {len(wps)} pts')

            # Activar/desactivar navegación
            elif t == 'activate':
                msg = Bool()
                msg.data = True
                self.pub_active.publish(msg)
                self.get_logger().info('Navegación activada desde HMI')

            elif t == 'deactivate':
                msg = Bool()
                msg.data = False
                self.pub_active.publish(msg)
                self.get_logger().info('Navegación desactivada desde HMI')

            elif t == 'estop':
                # Detener motores inmediatamente
                stop = Float32MultiArray()
                stop.data = [90.0, 0.0, 0.0, 0.0, 0.0]
                self.pub_pwm.publish(stop)
                msg = Bool()
                msg.data = False
                self.pub_active.publish(msg)
                self.get_logger().warn('E-STOP recibido')

        except json.JSONDecodeError:
            self.get_logger().warning(f'JSON inválido: {line}')

    def _teleop_watchdog(self):
        if time.time() - self._last_cmd_time > 1.0:
            msg = Bool()
            msg.data = False
            self.pub_teleop.publish(msg)

    def destroy_node(self):
        self._running = False
        if self.ser:
            self.ser.close()
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