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
import queue


class STM32Bridge(Node):
    def __init__(self):
        super().__init__('stm32_bridge')

        self.declare_parameter('port', '/dev/ttyAMA0')
        self.declare_parameter('baud', 115200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # Publishers → ROS
        self.pub_uf    = self.create_publisher(Float32, '/ultrasonic/front', 10)
        self.pub_ur    = self.create_publisher(Float32, '/ultrasonic/rear',  10)
        self.pub_imu_f = self.create_publisher(String,  '/imu/front',        10)
        self.pub_imu_r = self.create_publisher(String,  '/imu/rear',         10)

        # Subscribers
        self.create_subscription(Float32MultiArray, '/control/pwm_cmd',
            self.cb_pwm, 10)
        self.create_subscription(Bool, '/teleop/active',
            lambda m: setattr(self, '_teleop_active', m.data), 10)

        self._teleop_active = False
        self._last_cmd = {'s': 135.0, 'rl': 0.0, 'rr': 0.0, 'fl': 0.0, 'fr': 0.0}
        self._lock = threading.Lock()
        self._tx_queue = queue.Queue(maxsize=2)
        self._running = True

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'STM32: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(f'Error UART: {e}')
            self.ser = None

        # Timer TX 20ms
        self.create_timer(0.5, self._enqueue_cmd)

        # Threads
        threading.Thread(target=self._writer, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def cb_pwm(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            return

        with self._lock:
            self._last_cmd = {
                's':  round(float(msg.data[0]), 2),
                'rl': round(float(msg.data[1]), 2),
                'rr': round(float(msg.data[2]), 2),
                'fl': round(float(msg.data[3]), 2),
                'fr': round(float(msg.data[4]), 2),
            }

    # ── TX ────────────────────────────────────────────────────────────────────

    def _enqueue_cmd(self):
        with self._lock:
            cmd = dict(self._last_cmd)

        # Comparar con tolerancia para el servo, exacto para motores
        if self._sent_cmd is not None:
            servo_same = abs(cmd['s'] - self._sent_cmd['s']) < 10.0   # < 1° de diferencia
            motors_same = all(
                abs(cmd[k] - self._sent_cmd[k]) < 0.01
                for k in ('rl', 'rr', 'fl', 'fr')
            )
            if servo_same and motors_same:
                return

        self._sent_cmd = dict(cmd)
        ...

    def _writer(self):
        while self._running:
            try:
                cmd = self._tx_queue.get(timeout=0.1)
                if not self.ser or not self.ser.is_open:
                    continue
                try:
                    self.ser.write((json.dumps(cmd) + '\n').encode())
                    self.ser.flush()
                except Exception as e:
                    self.get_logger().error(f'TX: {e}')
            except queue.Empty:
                continue

    # ── RX ────────────────────────────────────────────────────────────────────

    def _reader(self):
        buf = ''
        while self._running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.05)
                continue
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    buf += self.ser.read(waiting).decode('utf-8', errors='ignore')

                    # Si el buffer crece demasiado descartar datos viejos
                    if len(buf) > 512:
                        last_nl = buf.rfind('\n', 0, -1)
                        if last_nl != -1:
                            buf = buf[last_nl + 1:]

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
                    time.sleep(0.002)
            except Exception as e:
                self.get_logger().error(f'RX: {e}')
                time.sleep(0.1)

    # ── Publicar telemetría ───────────────────────────────────────────────────

    def _publish_tel(self, obj):
        if 'uf' in obj:
            msg = Float32()
            msg.data = float(obj['uf'])
            self.pub_uf.publish(msg)

        if 'ur' in obj:
            msg = Float32()
            msg.data = float(obj['ur'])
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

    # ── Cleanup ───────────────────────────────────────────────────────────────

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