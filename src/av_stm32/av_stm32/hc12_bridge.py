#!/usr/bin/env python3
"""
HC-12 Bridge Node
Lee comandos JSON del HC-12 → publica en /control/pwm_cmd
Subscribe /ultrasonic/front y /ultrasonic/rear → envía telemetría por HC-12
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String
import serial
import json
import threading


class HC12Bridge(Node):
    def __init__(self):
        super().__init__('hc12_bridge')

        # Parámetros
        self.declare_parameter('port', '/dev/ttyUSB1')
        self.declare_parameter('baud', 9600)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # Publisher → STM32
        self.pub_pwm = self.create_publisher(
            Float32MultiArray, '/control/pwm_cmd', 10)

        # Subscribers → telemetría
        self.create_subscription(Float32, '/ultrasonic/front',
            lambda m: self._cache_tel('uf', m.data), 10)
        self.create_subscription(Float32, '/ultrasonic/rear',
            lambda m: self._cache_tel('ur', m.data), 10)

        self.create_subscription(String, '/imu/front',
            lambda m: self._cache_tel('imu_f', m.data), 10)
        self.create_subscription(String, '/imu/rear',
            lambda m: self._cache_tel('imu_r', m.data), 10)

        self.tel = {'uf': 0.0, 'ur': 0.0}

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

    def _cache_tel(self, key, val):
        self.tel[key] = round(float(val), 2)

    def _send_telemetry(self):
        if not self.ser or not self.ser.is_open:
            return
        try:
            msg = json.dumps({
                't':     'tel',
                'uf':    self.tel['uf'],
                'ur':    self.tel['ur'],
                'imu_f': self.tel['imu_f'],
                'imu_r': self.tel['imu_r']
            }) + '\n'
            self.ser.write(msg.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f'Error TX: {e}')

    def _reader(self):
        buf = ''
        while self._running:
            if not self.ser or not self.ser.is_open:
                import time; time.sleep(0.1); continue
            try:
                data = self.ser.read(256).decode('utf-8', errors='ignore')
                if data:
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._parse_cmd(line)
            except Exception as e:
                self.get_logger().error(f'Error RX: {e}')

    def _parse_cmd(self, line):
        try:
            obj = json.loads(line)
            if obj.get('t') != 'cmd':
                return

            msg = Float32MultiArray()
            msg.data = [
                float(obj.get('s',  0.0)),  # servo
                float(obj.get('rl', 0.0)),  # RL
                float(obj.get('rr', 0.0)),  # RR
                float(obj.get('fl', 0.0)),  # FL
                float(obj.get('fr', 0.0)),  # FR
            ]
            self.pub_pwm.publish(msg)
            self.get_logger().debug(f'CMD: {msg.data}')

        except json.JSONDecodeError:
            self.get_logger().warning(f'JSON inválido: {line}')

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
