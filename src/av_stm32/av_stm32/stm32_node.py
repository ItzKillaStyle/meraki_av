import struct
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Bool
from av_interfaces.msg import VehicleState
from std_msgs.msg import Float32MultiArray
import serial

# ── Protocolo UART ────────────────────────────────────────────────────────────
# RPi5 → STM32  CMD      : [0xAA, 0x01, int16 steering, int16 m12, int16 m34, uint8 flags, uint8 crc, 0x55]  = 10 bytes
# STM32 → RPi5  TELEMETRY: [0xBB, 0x02, float32×4 imu1, float32×4 imu2, uint16 us_front, uint16 us_rear, uint8 status, uint8 crc, 0x55] = 40 bytes

CMD_START  = 0xAA
CMD_TYPE   = 0x01
TEL_START  = 0xBB
TEL_TYPE   = 0x02
END_BYTE   = 0x55

CMD_LEN = 10   # bytes totales trama CMD
TEL_LEN = 40   # bytes totales trama TELEMETRY

# Índices dentro de la trama TELEMETRY (sin el byte START)
TEL_PAYLOAD_LEN = TEL_LEN - 2  # sin START y END


def crc8(data: bytes) -> int:
    """CRC-8 Dallas/Maxim — igual en STM32 y RPi5."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


def int16_to_bytes(value: float) -> bytes:
    """Convierte -1.0..1.0 → int16 → 2 bytes big-endian."""
    v = max(-1.0, min(1.0, value))
    i = int(v * 32767)
    return struct.pack('>h', i)


class Stm32BridgeNode(Node):

    def __init__(self):
        super().__init__('stm32_bridge_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('port',          '/dev/ttyAMA0')
        self.declare_parameter('baudrate',      460800)
        self.declare_parameter('cmd_rate_hz',   50.0)
        self.declare_parameter('watchdog_s',    0.5)
        self.declare_parameter('frame_id_front', 'imu_front_link')
        self.declare_parameter('frame_id_rear',  'imu_rear_link')

        self.port            = self.get_parameter('port').value
        self.baudrate        = self.get_parameter('baudrate').value
        self.watchdog_s      = self.get_parameter('watchdog_s').value
        self.frame_id_front  = self.get_parameter('frame_id_front').value
        self.frame_id_rear   = self.get_parameter('frame_id_rear').value

        # ── Estado interno ────────────────────────────────────────────────────
        self.steering  = 0.0   # -1.0 .. 1.0
        self.motor_12  = 0.0   # -1.0 .. 1.0  (ruedas izquierda)
        self.motor_34  = 0.0   # -1.0 .. 1.0  (ruedas derecha)
        self.flags     = 0x00  # bit0=estop bit1=luces bit2=bocina
        self.emergency = False
        self.lock      = threading.Lock()
        self.last_cmd  = self.get_clock().now()

        # ── Serial ────────────────────────────────────────────────────────────
        try:
            self.serial = serial.Serial(
                self.port, self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1)
            self.get_logger().info(
                f'STM32 bridge conectado — {self.port} @ {self.baudrate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'No se pudo abrir {self.port}: {e}')
            self.serial = None

        # ── Subscribers ───────────────────────────────────────────────────────
        # Comando de control: [steering, motor_12, motor_34]  (-1.0..1.0)
        self.create_subscription(
            Float32MultiArray, '/control/pwm_cmd', self.cb_pwm_cmd, 10)

        # Parada de emergencia
        self.create_subscription(
            Bool, '/emergency_stop', self.cb_estop, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_imu_front  = self.create_publisher(Imu,   '/imu/front',          10)
        self.pub_imu_rear   = self.create_publisher(Imu,   '/imu/rear',           10)
        self.pub_us_front   = self.create_publisher(Range, '/ultrasonic/front',   10)
        self.pub_us_rear    = self.create_publisher(Range, '/ultrasonic/rear',    10)
        self.pub_state      = self.create_publisher(
            VehicleState, '/vehicle/state', 10)

        # ── Timers ────────────────────────────────────────────────────────────
        period = 1.0 / self.get_parameter('cmd_rate_hz').value
        self.create_timer(period,       self.send_cmd)
        self.create_timer(self.watchdog_s / 5, self.watchdog_cb)

        # ── Hilo lector ───────────────────────────────────────────────────────
        self.read_thread = threading.Thread(
            target=self.read_loop, daemon=True)
        self.read_thread.start()

        self.get_logger().info('STM32 bridge iniciado — 50 Hz TX | hilo RX activo')

    # ── Callbacks subscribers ─────────────────────────────────────────────────

    def cb_pwm_cmd(self, msg: Float32MultiArray):
        """Recibe [steering, motor_12, motor_34] de av_control."""
        if self.emergency:
            return
        if len(msg.data) != 3:
            self.get_logger().warn(
                f'pwm_cmd debe tener 3 valores [steering, m12, m34], recibí {len(msg.data)}')
            return
        with self.lock:
            self.steering = float(msg.data[0])
            self.motor_12 = float(msg.data[1])
            self.motor_34 = float(msg.data[2])
            self.last_cmd = self.get_clock().now()

    def cb_estop(self, msg: Bool):
        self.emergency = msg.data
        if self.emergency:
            with self.lock:
                self.steering = 0.0
                self.motor_12 = 0.0
                self.motor_34 = 0.0
                self.flags    = self.flags | 0x01  # bit0 = estop
            self.get_logger().warn('EMERGENCIA ACTIVA — motores detenidos')
        else:
            with self.lock:
                self.flags = self.flags & ~0x01
            self.get_logger().info('Emergencia desactivada')

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def watchdog_cb(self):
        """Si no llega comando en watchdog_s segundos, detiene motores."""
        elapsed = (self.get_clock().now() - self.last_cmd).nanoseconds / 1e9
        if elapsed > self.watchdog_s:
            with self.lock:
                self.motor_12 = 0.0
                self.motor_34 = 0.0

    # ── TX: envío de trama CMD a STM32 ───────────────────────────────────────

    def send_cmd(self):
        if self.serial is None or not self.serial.is_open:
            return

        with self.lock:
            s  = self.steering
            m12 = self.motor_12
            m34 = self.motor_34
            f  = self.flags

        payload = (
            bytes([CMD_TYPE]) +
            int16_to_bytes(s)   +
            int16_to_bytes(m12) +
            int16_to_bytes(m34) +
            bytes([f])
        )
        crc     = crc8(payload)
        packet  = bytes([CMD_START]) + payload + bytes([crc, END_BYTE])

        try:
            self.serial.write(packet)
        except serial.SerialException as e:
            self.get_logger().error(f'Error TX serial: {e}')

    # ── RX: lectura continua de telemetría desde STM32 ────────────────────────

    def read_loop(self):
        """Hilo dedicado — busca byte START 0xBB y lee trama completa."""
        while rclpy.ok():
            if self.serial is None or not self.serial.is_open:
                continue
            try:
                byte = self.serial.read(1)
                if not byte or byte[0] != TEL_START:
                    continue

                # Leer resto de la trama
                rest = self.serial.read(TEL_LEN - 1)
                if len(rest) != TEL_LEN - 1:
                    continue

                frame = bytes([TEL_START]) + rest

                # Verificar END
                if frame[-1] != END_BYTE:
                    continue

                # Verificar TYPE
                if frame[1] != TEL_TYPE:
                    continue

                # Verificar CRC (sobre todo excepto START, CRC y END)
                payload  = frame[1:-2]
                crc_recv = frame[-2]
                if crc8(payload) != crc_recv:
                    self.get_logger().warn('CRC error en trama TELEMETRY')
                    continue

                self.parse_telemetry(frame)

            except serial.SerialException as e:
                self.get_logger().error(f'Error RX serial: {e}')
            except Exception as e:
                self.get_logger().debug(f'read_loop excepción: {e}')

    def parse_telemetry(self, frame: bytes):
        """
        Trama TELEMETRY layout (40 bytes):
        [0]      START  0xBB
        [1]      TYPE   0x02
        [2-17]   BNO055 #1: w, x, y, z  (4× float32 BE)
        [18-33]  BNO055 #2: w, x, y, z  (4× float32 BE)
        [34-35]  HC-SR04 front  (uint16 BE, cm)
        [36-37]  HC-SR04 rear   (uint16 BE, cm)
        [38]     status
        [39-38]  CRC8  (índice 38)
        [39]     END   0x55
        """
        now = self.get_clock().now().to_msg()

        # ── BNO055 #1 ─────────────────────────────────────────────────────────
        imu1_w, imu1_x, imu1_y, imu1_z = struct.unpack_from('>ffff', frame, 2)
        imu_front               = Imu()
        imu_front.header.stamp  = now
        imu_front.header.frame_id = self.frame_id_front
        imu_front.orientation.w = float(imu1_w)
        imu_front.orientation.x = float(imu1_x)
        imu_front.orientation.y = float(imu1_y)
        imu_front.orientation.z = float(imu1_z)
        # Covarianza desconocida → -1 en diagonal (ROS2 convención)
        imu_front.orientation_covariance[0] = -1.0
        self.pub_imu_front.publish(imu_front)

        # ── BNO055 #2 ─────────────────────────────────────────────────────────
        imu2_w, imu2_x, imu2_y, imu2_z = struct.unpack_from('>ffff', frame, 18)
        imu_rear                = Imu()
        imu_rear.header.stamp   = now
        imu_rear.header.frame_id = self.frame_id_rear
        imu_rear.orientation.w  = float(imu2_w)
        imu_rear.orientation.x  = float(imu2_x)
        imu_rear.orientation.y  = float(imu2_y)
        imu_rear.orientation.z  = float(imu2_z)
        imu_rear.orientation_covariance[0] = -1.0
        self.pub_imu_rear.publish(imu_rear)

        # ── HC-SR04 ───────────────────────────────────────────────────────────
        us_front_cm, us_rear_cm = struct.unpack_from('>HH', frame, 34)

        range_front                   = Range()
        range_front.header.stamp      = now
        range_front.header.frame_id   = 'ultrasonic_front_link'
        range_front.radiation_type    = Range.ULTRASOUND
        range_front.field_of_view     = 0.26   # ~15 grados HC-SR04
        range_front.min_range         = 0.02
        range_front.max_range         = 4.00
        range_front.range             = float(us_front_cm) / 100.0
        self.pub_us_front.publish(range_front)

        range_rear                    = Range()
        range_rear.header.stamp       = now
        range_rear.header.frame_id    = 'ultrasonic_rear_link'
        range_rear.radiation_type     = Range.ULTRASOUND
        range_rear.field_of_view      = 0.26
        range_rear.min_range          = 0.02
        range_rear.max_range          = 4.00
        range_rear.range              = float(us_rear_cm) / 100.0
        self.pub_us_rear.publish(range_rear)

        # ── VehicleState ──────────────────────────────────────────────────────
        status = frame[38]
        state                   = VehicleState()
        state.header.stamp      = now
        state.header.frame_id   = 'base_link'
        state.emergency_stop    = bool(status & 0x01)
        state.steering_angle_rad = 0.0   # sin encoder de servo por ahora
        state.speed_ms           = 0.0   # sin encoders de rueda
        self.pub_state.publish(state)

        self.get_logger().debug(
            f'TEL | imu1=({imu1_w:.2f},{imu1_x:.2f},{imu1_y:.2f},{imu1_z:.2f}) '
            f'us_f={us_front_cm}cm us_r={us_rear_cm}cm status=0x{status:02X}'
        )

    def destroy_node(self):
        if self.serial and self.serial.is_open:
            # Manda stop antes de cerrar
            with self.lock:
                self.motor_12 = 0.0
                self.motor_34 = 0.0
            self.send_cmd()
            self.serial.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Stm32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()