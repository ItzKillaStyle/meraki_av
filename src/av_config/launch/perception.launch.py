import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def cfg(package: str, yaml: str) -> str:
    return os.path.join(
        get_package_share_directory(package), 'config', yaml)


def generate_launch_description():

    args = [
        DeclareLaunchArgument('debug', default_value='true',
                              description='Publica imagen debug'),
    ]

    debug = LaunchConfiguration('debug')

    return LaunchDescription(args + [
        Node(package='av_camera',   executable='camera_node',
             parameters=[cfg('av_camera', 'camera.yaml')],
             output='screen'),

        Node(package='av_vision',   executable='vision_node',
             parameters=[cfg('av_vision', 'vision.yaml'), {'debug': debug}],
             output='screen'),

        Node(package='av_lidar',    executable='lidar_node',
             parameters=[cfg('av_lidar', 'lidar.yaml')],
             output='screen'),

        Node(package='av_obstacle', executable='obstacle_node',
             parameters=[cfg('av_obstacle', 'obstacle.yaml')],
             output='screen'),
    ])