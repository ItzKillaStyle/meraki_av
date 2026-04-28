import serial
import pynmea2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import TwistWithCovarianceStamped
from std_msgs.msg import Float64, UInt8


class GpsNode(Node):

    def __init__(self):
        super().__init__('gps_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('port',      '/dev/ttyGPS')
        self.declare_parameter('baudrate',  9600)
        self.declare_parameter('frame_id',  'gps_link')
        self.declare_parameter('read_hz',   10.0)

        self.port     = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value

        # ── Estado interno ────────────────────────────────────────────────────
        # GGA → posición + calidad
        self._lat      = 0.0
        self._lon      = 0.0
        self._alt      = 0.0
        self._sats     = 0
        self._fix_qual = 0   # 0=no fix, 1=GPS, 2=DGPS

        # RMC → velocidad + heading
        self._speed_ms  = 0.0   # m/s
        self._heading   = 0.0   # grados 0-360

        # VTG → velocidad más precisa
        self._speed_vtg = 0.0

        self._has_fix = False

        # ── Serial ────────────────────────────────────────────────────────────
        try:
            self.serial = serial.Serial(
                self.port, self.baudrate,
                timeout=1.0)
            self.get_logger().info(
                f'GPS NEO-8M conectado — {self.port} @ {self.baudrate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'No se pudo abrir {self.port}: {e}')
            self.serial = None

        # ── Publishers ────────────────────────────────────────────────────────
        # Posición principal — usado por robot_localization EKF
        self.pub_fix = self.create_publisher(
            NavSatFix, '/gps/fix', 10)

        # Velocidad sobre suelo en frame gps_link
        self.pub_vel = self.create_publisher(
            TwistWithCovarianceStamped, '/gps/vel', 10)

        # Heading (rumbo respecto al norte) en grados
        self.pub_heading = self.create_publisher(
            Float64, '/gps/heading', 10)

        # Número de satélites — útil para diagnóstico
        self.pub_sats = self.create_publisher(
            UInt8, '/gps/satellites', 10)

        # ── Timer de lectura ──────────────────────────────────────────────────
        period = 1.0 / self.get_parameter('read_hz').value
        self.create_timer(period, self.read_cb)

        self.get_logger().info('GPS node iniciado — esperando fix...')

    # ── Timer callback: lee y parsea líneas NMEA ──────────────────────────────

    def read_cb(self):
        if self.serial is None or not self.serial.is_open:
            return

        # Lee todas las líneas disponibles en el buffer en este ciclo
        lines_read = 0
        while self.serial.in_waiting and lines_read < 20:
            try:
                raw = self.serial.readline()
                line = raw.decode('ascii', errors='replace').strip()
                lines_read += 1

                if not line.startswith('$'):
                    continue

                self.parse_nmea(line)

            except serial.SerialException as e:
                self.get_logger().error(f'Error leyendo serial: {e}')
                break
            except Exception as e:
                self.get_logger().debug(f'Error parseando línea: {e}')

    # ── Parser NMEA ───────────────────────────────────────────────────────────

    def parse_nmea(self, line: str):
        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return

        sentence = type(msg).__name__

        # ── GGA: posición, altitud, calidad, satélites ────────────────────────
        if sentence == 'GGA':
            if msg.gps_qual is None or msg.gps_qual == 0:
                self._has_fix = False
                return

            self._has_fix  = True
            self._lat      = float(msg.latitude)
            self._lon      = float(msg.longitude)
            self._alt      = float(msg.altitude) if msg.altitude else 0.0
            self._fix_qual = int(msg.gps_qual)
            self._sats     = int(msg.num_sats) if msg.num_sats else 0

            self.publish_fix()
            self.publish_sats()

        # ── RMC: velocidad sobre suelo + heading ──────────────────────────────
        elif sentence == 'RMC':
            if not msg.status or msg.status != 'A':
                return  # A = datos válidos

            # Velocidad: knots → m/s
            if msg.spd_over_grnd is not None:
                self._speed_ms = float(msg.spd_over_grnd) * 0.51444

            # Heading verdadero
            if msg.true_course is not None:
                self._heading = float(msg.true_course)

            self.publish_velocity()
            self.publish_heading()

        # ── VTG: velocidad más precisa que RMC ────────────────────────────────
        elif sentence == 'VTG':
            if msg.spd_over_grnd_kmph is not None:
                # km/h → m/s
                self._speed_vtg = float(msg.spd_over_grnd_kmph) / 3.6
                self._speed_ms  = self._speed_vtg

            if msg.true_track is not None:
                self._heading = float(msg.true_track)

    # ── Publicadores ─────────────────────────────────────────────────────────

    def publish_fix(self):
        fix = NavSatFix()
        fix.header.stamp    = self.get_clock().now().to_msg()
        fix.header.frame_id = self.frame_id

        fix.latitude  = self._lat
        fix.longitude = self._lon
        fix.altitude  = self._alt

        # Estado del fix
        fix.status.service = NavSatStatus.SERVICE_GPS
        if self._fix_qual == 0:
            fix.status.status = NavSatStatus.STATUS_NO_FIX
        elif self._fix_qual == 2:
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX  # DGPS
        else:
            fix.status.status = NavSatStatus.STATUS_FIX

        # Covarianza diagonal NEO-8M:
        # CEP ~2.5m horizontal → varianza ~2.5² = 6.25 m²
        # Vertical ~2× horizontal → ~12.5 m²
        cov = 6.25 if self._fix_qual >= 1 else 999.0
        fix.position_covariance = [
            cov,  0.0,  0.0,
            0.0,  cov,  0.0,
            0.0,  0.0,  cov * 2.0
        ]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.pub_fix.publish(fix)

        self.get_logger().debug(
            f'FIX | lat={self._lat:.6f} lon={self._lon:.6f} '
            f'alt={self._alt:.1f}m sats={self._sats} qual={self._fix_qual}'
        )

    def publish_velocity(self):
        vel = TwistWithCovarianceStamped()
        vel.header.stamp    = self.get_clock().now().to_msg()
        vel.header.frame_id = self.frame_id

        # Velocidad lineal en X (hacia adelante en frame del GPS)
        vel.twist.twist.linear.x = self._speed_ms
        vel.twist.twist.linear.y = 0.0
        vel.twist.twist.linear.z = 0.0

        # Covarianza velocidad NEO-8M ~0.1 m/s → varianza 0.01
        vel.twist.covariance[0]  = 0.01   # vx
        vel.twist.covariance[7]  = 0.01   # vy
        vel.twist.covariance[14] = 0.01   # vz

        self.pub_vel.publish(vel)

    def publish_heading(self):
        msg = Float64()
        msg.data = self._heading
        self.pub_heading.publish(msg)

    def publish_sats(self):
        msg = UInt8()
        msg.data = self._sats
        self.pub_sats.publish(msg)

    def destroy_node(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()