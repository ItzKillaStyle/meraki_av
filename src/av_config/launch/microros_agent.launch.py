from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument('port',     default_value='/dev/ttyAMA0'),
        DeclareLaunchArgument('baudrate', default_value='460800'),

        Node(
            package='micro_ros_agent',
            executable='micro_ros_agent',
            name='micro_ros_agent',
            arguments=[
                'serial',
                '--dev', LaunchConfiguration('port'),
                '-b',    LaunchConfiguration('baudrate'),
                '-v4'    # verbose para debug, cámbialo a '' en producción
            ],
            output='screen'
        )
    ])
