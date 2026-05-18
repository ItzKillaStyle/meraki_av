#!/usr/bin/env python3
"""
orientation_node.py

Fusiona:
- IMU frontal
- IMU trasera
- GPS

Calcula:
- heading/yaw
- roll
- pitch

Publica:
- /vehicle/heading
- /vehicle/roll
- /vehicle/pitch
"""

import json
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from sensor_msgs.msg import NavSatFix


EARTH_RADIUS_M = 6371000.0


def bearing(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlon = lon2 - lon1

    x = (
        math.sin(dlon) *
        math.cos(lat2)
    )

    y = (
        math.cos(lat1) * math.sin(lat2) -
        math.sin(lat1) * math.cos(lat2) *
        math.cos(dlon)
    )

    return math.atan2(x, y)


class OrientationNode(Node):

    def __init__(self):

        super().__init__('orientation_node')

        # ---------------------------------------------------------
        # Parámetros
        # ---------------------------------------------------------

        self.declare_parameter('hz', 50.0)

        self.declare_parameter(
            'gyro_alpha',
            0.98
        )

        self.declare_parameter(
            'gps_correction_gain',
            0.02
        )

        self.declare_parameter(
            'gps_min_distance',
            1.0
        )

        self.hz = self.get_parameter('hz').value

        self.gyro_alpha = self.get_parameter(
            'gyro_alpha'
        ).value

        self.gps_gain = self.get_parameter(
            'gps_correction_gain'
        ).value

        self.gps_min_distance = self.get_parameter(
            'gps_min_distance'
        ).value

        # ---------------------------------------------------------
        # Estado IMU
        # ---------------------------------------------------------

        self.imu_front = {}
        self.imu_rear = {}

        # orientación
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # GPS
        self.prev_lat = None
        self.prev_lon = None

        self.current_lat = None
        self.current_lon = None

        # tiempo
        self.last_time = self.get_clock().now()

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        self.create_subscription(
            String,
            '/imu/front',
            self.cb_imu_front,
            10
        )

        self.create_subscription(
            String,
            '/imu/rear',
            self.cb_imu_rear,
            10
        )

        self.create_subscription(
            NavSatFix,
            '/gps/fix',
            self.cb_gps,
            10
        )

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------

        self.pub_heading = self.create_publisher(
            Float32,
            '/vehicle/heading',
            10
        )

        self.pub_roll = self.create_publisher(
            Float32,
            '/vehicle/roll',
            10
        )

        self.pub_pitch = self.create_publisher(
            Float32,
            '/vehicle/pitch',
            10
        )

        # ---------------------------------------------------------
        # Timer
        # ---------------------------------------------------------

        self.create_timer(
            1.0 / self.hz,
            self.update
        )

        self.get_logger().info(
            'Orientation node iniciado'
        )

    # =============================================================
    # IMU callbacks
    # =============================================================

    def cb_imu_front(self, msg):

        try:
            self.imu_front = json.loads(msg.data)
        except Exception:
            pass

    def cb_imu_rear(self, msg):

        try:
            self.imu_rear = json.loads(msg.data)
        except Exception:
            pass

    # =============================================================
    # GPS callback
    # =============================================================

    def cb_gps(self, msg):

        if msg.status.status < 0:
            return

        lat = msg.latitude
        lon = msg.longitude

        if self.current_lat is not None:

            dist = self._gps_distance(
                self.current_lat,
                self.current_lon,
                lat,
                lon
            )

            # -----------------------------------------------------
            # Corrección heading usando GPS
            # -----------------------------------------------------

            if dist >= self.gps_min_distance:

                gps_heading = bearing(
                    self.current_lat,
                    self.current_lon,
                    lat,
                    lon
                )

                yaw_error = (
                    gps_heading -
                    self.yaw
                )

                yaw_error = math.atan2(
                    math.sin(yaw_error),
                    math.cos(yaw_error)
                )

                self.yaw += (
                    yaw_error *
                    self.gps_gain
                )

                self.yaw = math.atan2(
                    math.sin(self.yaw),
                    math.cos(self.yaw)
                )

        self.prev_lat = self.current_lat
        self.prev_lon = self.current_lon

        self.current_lat = lat
        self.current_lon = lon

    # =============================================================
    # Main update
    # =============================================================

    def update(self):

        now = self.get_clock().now()

        dt = (
            now -
            self.last_time
        ).nanoseconds / 1e9

        self.last_time = now

        if dt <= 0.0:
            return

        if not self.imu_front:
            return

        # ---------------------------------------------------------
        # Fusionar IMUs
        # ---------------------------------------------------------

        gx = self._avg('gx')
        gy = self._avg('gy')
        gz = self._avg('gz')

        gz = math.radians(gz)
        gx = math.radians(gx)
        gy = math.radians(gy)

        ax = self._avg('ax')
        ay = self._avg('ay')
        az = self._avg('az')

        # ---------------------------------------------------------
        # Roll y pitch desde acelerómetro
        # ---------------------------------------------------------

        accel_roll = math.atan2(
            ay,
            az
        )

        accel_pitch = math.atan2(
            -ax,
            math.sqrt(
                ay * ay +
                az * az
            )
        )

        # ---------------------------------------------------------
        # Integrar gyro
        # ---------------------------------------------------------

        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        # ---------------------------------------------------------
        # Complementary filter
        # ---------------------------------------------------------

        alpha = self.gyro_alpha

        self.roll = (
            alpha * self.roll +
            (1.0 - alpha) * accel_roll
        )

        self.pitch = (
            alpha * self.pitch +
            (1.0 - alpha) * accel_pitch
        )

        # normalizar yaw
        self.yaw = math.atan2(
            math.sin(self.yaw),
            math.cos(self.yaw)
        )

        # ---------------------------------------------------------
        # Publicar
        # ---------------------------------------------------------

        heading_msg = Float32()
        heading_msg.data = float(self.yaw)

        roll_msg = Float32()
        roll_msg.data = float(self.roll)

        pitch_msg = Float32()
        pitch_msg.data = float(self.pitch)

        self.pub_heading.publish(
            heading_msg
        )

        self.pub_roll.publish(
            roll_msg
        )

        self.pub_pitch.publish(
            pitch_msg
        )

    # =============================================================
    # Helpers
    # =============================================================

    def _avg(self, key):

        vals = []

        if key in self.imu_front:
            vals.append(
                float(self.imu_front[key])
            )

        if key in self.imu_rear:
            vals.append(
                float(self.imu_rear[key])
            )

        if not vals:
            return 0.0

        return sum(vals) / len(vals)

    def _gps_distance(
        self,
        lat1,
        lon1,
        lat2,
        lon2
    ):

        lat1, lon1, lat2, lon2 = map(
            math.radians,
            [lat1, lon1, lat2, lon2]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1) *
            math.cos(lat2) *
            math.sin(dlon / 2) ** 2
        )

        return (
            2.0 *
            EARTH_RADIUS_M *
            math.asin(math.sqrt(a))
        )


def main(args=None):

    rclpy.init(args=args)

    node = OrientationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()