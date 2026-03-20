from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg = get_package_share_directory('av_localization')

    ekf_cfg    = os.path.join(pkg, 'config', 'ekf.yaml')
    navsat_cfg = os.path.join(pkg, 'config', 'navsat.yaml')
    loc_cfg    = os.path.join(pkg, 'config', 'localization.yaml')

    return LaunchDescription([

        # 1. Nodo propio — fusiona BNO055 x2 y publica /imu/fused
        Node(
            package='av_localization',
            executable='localization_node',
            name='localization_node',
            parameters=[loc_cfg],
            output='screen'
        ),

        # 2. EKF — fusiona /imu/fused + /odometry/gps → /odometry/filtered
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[ekf_cfg],
            remappings=[('odometry/filtered', '/odometry/filtered')],
            output='screen'
        ),

        # 3. navsat_transform — convierte /gps/fix → /odometry/gps
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            parameters=[navsat_cfg],
            remappings=[
                ('imu/data',            '/imu/fused'),
                ('gps/fix',             '/gps/fix'),
                ('odometry/filtered',   '/odometry/filtered'),
            ],
            output='screen'
        ),
    ])