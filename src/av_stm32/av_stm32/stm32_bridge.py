#!/usr/bin/env python3
"""
STM32 UART Bridge
UART ↔ ROS2
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String, Bool
import serial
import json
import threading
import time


class STM32Bridge(Node):
    def __init__(self):
        super().__init__('stm32_bridge')

        self.declare_parameter('port', '/dev/ttyAMA0')
        self.declare_parameter('baud', 115200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # Publishers → ROS (datos que vienen del STM32)
        self.pub_uf     = self.create_publisher(Float32, '/ultrasonic/front', 10)
        self.pub_ur     = self.create_publisher(Float32, '/ultrasonic/rear',  10)
        self.pub_imu_f  = self.create_publisher(String,  '/imu/front',        10)
        self.pub_imu_r  = self.create_publisher(String,  '/imu/rear',         10)

        # Subscriber → recibe comandos PWM y los manda al STM32
        self.create_subscription(Float32MultiArray, '/control/pwm_cmd',
            self.cb_pwm, 10)

        # Subscriber → teleop activo
        self._teleop_active = False
        self.create_subscription(Bool, '/teleop/active',
            lambda m: setattr(self, '_teleop_active', m.data), 10)

        self._last_cmd = {'s': 135.0, 'rl': 0.0, 'rr': 0.0, 'fl': 0.0, 'fr': 0.0}
        self._lock = threading.Lock()

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'STM32 conectado: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(f'Error UART: {e}')
            self.ser = None

        # Timer TX → envía comando al STM32 cada 50ms
        self.create_timer(0.05, self._send_cmd)

        # Thread RX → lee telemetría del STM32
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def cb_pwm(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            return
        with self._lock:
            self._last_cmd = {
                's':  float(msg.data[0]),
                'rl': float(msg.data[1]),
                'rr': float(msg.data[2]),
                'fl': float(msg.data[3]),
                'fr': float(msg.data[4]),
            }

    def _send_cmd(self):
        if not self.ser or not self.ser.is_open:
            return
        with self._lock:
            cmd = dict(self._last_cmd)
        try:
            self.ser.write((json.dumps(cmd) + '\n').encode())
        except Exception as e:
            self.get_logger().error(f'TX error: {e}')

    def _reader(self):
        buf = ''
        while self._running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    buf += self.ser.read(waiting).decode('utf-8', errors='ignore')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            self._publish_tel(obj)
                        except json.JSONDecodeError:
                            pass
                else:
                    time.sleep(0.005)
            except Exception as e:
                self.get_logger().error(f'RX error: {e}')
                time.sleep(0.1)

    def _publish_tel(self, obj):
        if 'uf' in obj:
            msg = Float32(); msg.data = float(obj['uf'])
            self.pub_uf.publish(msg)

        if 'ur' in obj:
            msg = Float32(); msg.data = float(obj['ur'])
            self.pub_ur.publish(msg)

        if 'imu_f' in obj:
            msg = String()
            imu = obj['imu_f']
            msg.data = json.dumps(imu) if isinstance(imu, dict) else str(imu)
            self.pub_imu_f.publish(msg)

        if 'imu_r' in obj:
            msg = String()
            imu = obj['imu_r']
            msg.data = json.dumps(imu) if isinstance(imu, dict) else str(imu)
            self.pub_imu_r.publish(msg)

    def destroy_node(self):
        self._running = False
        if self.ser:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = STM32Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()