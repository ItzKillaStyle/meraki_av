import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion


def quat_to_array(q) -> np.ndarray:
    return np.array([q.w, q.x, q.y, q.z])


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Interpolación esférica entre dos quaterniones."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    if dot < 0:
        q2  = -q2
        dot = -dot
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    return (np.sin(theta_0 - theta) / sin_t0) * q1 + (np.sin(theta) / sin_t0) * q2


def array_to_quat_msg(q: np.ndarray) -> Quaternion:
    msg   = Quaternion()
    msg.w = float(q[0])
    msg.x = float(q[1])
    msg.y = float(q[2])
    msg.z = float(q[3])
    return msg


class LocalizationNode(Node):

    def __init__(self):
        super().__init__('localization_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('frame_id',       'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('imu_weight',     0.5)   # peso IMU frontal vs trasera
        self.declare_parameter('publish_hz',     50.0)

        self.frame_id       = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.imu_weight     = self.get_parameter('imu_weight').value
        self.publish_hz     = self.get_parameter('publish_hz').value

        # ── Estado interno ────────────────────────────────────────────────────
        self.imu_front_q  = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.imu_rear_q   = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_front_ok = False
        self.imu_rear_ok  = False
        self.gps_fix      = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Imu, '/imu/front', self.cb_imu_front, 10)
        self.create_subscription(
            Imu, '/imu/rear',  self.cb_imu_rear,  10)
        self.create_subscription(
            NavSatFix, '/gps/fix', self.cb_gps, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        # IMU fusionada — entrada al EKF de robot_localization
        self.pub_imu_fused = self.create_publisher(
            Imu, '/imu/fused', 10)

        # Pose con covarianza para robot_localization
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/pose', 10)

        # Odometría publicada por este nodo (complementa robot_localization)
        self.pub_odom = self.create_publisher(
            Odometry, '/localization/odom', 10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.publish_hz, self.publish_cb)

        self.get_logger().info(
            f'Localization node iniciado — '
            f'fusión BNO055 x2 | peso_frontal={self.imu_weight}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_imu_front(self, msg: Imu):
        q = msg.orientation
        norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        if norm < 0.1:
            return
        self.imu_front_q  = np.array([q.w, q.x, q.y, q.z]) / norm
        self.imu_front_ok = True

    def cb_imu_rear(self, msg: Imu):
        q = msg.orientation
        norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        if norm < 0.1:
            return
        self.imu_rear_q  = np.array([q.w, q.x, q.y, q.z]) / norm
        self.imu_rear_ok = True

    def cb_gps(self, msg: NavSatFix):
        self.gps_fix = msg

    # ── Publicación fusionada ─────────────────────────────────────────────────

    def publish_cb(self):
        now = self.get_clock().now().to_msg()

        # ── Fusión de quaterniones (SLERP) ────────────────────────────────────
        if self.imu_front_ok and self.imu_rear_ok:
            q_fused = slerp(
                self.imu_front_q,
                self.imu_rear_q,
                1.0 - self.imu_weight   # weight=0.5 → promedio exacto
            )
            source = 'front+rear'
        elif self.imu_front_ok:
            q_fused = self.imu_front_q
            source  = 'front only'
        elif self.imu_rear_ok:
            q_fused = self.imu_rear_q
            source  = 'rear only'
        else:
            return  # sin datos IMU — no publicar

        # Normaliza resultado
        q_fused = q_fused / np.linalg.norm(q_fused)

        # ── Publica IMU fusionada → robot_localization ────────────────────────
        imu_msg                  = Imu()
        imu_msg.header.stamp     = now
        imu_msg.header.frame_id  = 'imu_link'
        imu_msg.orientation      = array_to_quat_msg(q_fused)

        # Covarianza orientación BNO055 fusionado ~0.01 rad²
        cov = 0.01
        imu_msg.orientation_covariance = [
            cov,  0.0,  0.0,
            0.0,  cov,  0.0,
            0.0,  0.0,  cov
        ]
        # Sin acelerómetro/giróscopo por ahora — indicamos con -1
        imu_msg.angular_velocity_covariance[0]    = -1.0
        imu_msg.linear_acceleration_covariance[0] = -1.0

        self.pub_imu_fused.publish(imu_msg)

        # ── Publica Odometry básica ────────────────────────────────────────────
        odom                            = Odometry()
        odom.header.stamp               = now
        odom.header.frame_id            = self.frame_id
        odom.child_frame_id             = self.child_frame_id
        odom.pose.pose.orientation      = array_to_quat_msg(q_fused)
        odom.pose.pose.position.x       = 0.0  # actualizado por robot_localization
        odom.pose.pose.position.y       = 0.0
        odom.pose.pose.position.z       = 0.0

        # Covarianza diagonal pose
        odom.pose.covariance[0]  = 0.1   # x
        odom.pose.covariance[7]  = 0.1   # y
        odom.pose.covariance[35] = 0.05  # yaw

        self.pub_odom.publish(odom)

        self.get_logger().debug(
            f'IMU fused [{source}] | '
            f'q=({q_fused[0]:.3f},{q_fused[1]:.3f},'
            f'{q_fused[2]:.3f},{q_fused[3]:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()