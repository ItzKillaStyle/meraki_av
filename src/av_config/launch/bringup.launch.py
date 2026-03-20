import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def pkg(name: str) -> str:
    return get_package_share_directory(name)


def cfg(package: str, yaml: str) -> str:
    return os.path.join(pkg(package), 'config', yaml)


def generate_launch_description():

    # ── Argumentos configurables desde línea de comandos ─────────────────────
    args = [
        DeclareLaunchArgument('debug',
            default_value='false',
            description='Publica topics de debug (imagen procesada)'),
        DeclareLaunchArgument('use_gps',
            default_value='true',
            description='Habilita driver GPS NEO-8M'),
        DeclareLaunchArgument('use_lidar',
            default_value='true',
            description='Habilita driver LiDAR S2L'),
        DeclareLaunchArgument('use_stm32',
            default_value='true',
            description='Habilita bridge STM32'),
        DeclareLaunchArgument('use_localization',
            default_value='true',
            description='Habilita stack de localización EKF'),
        DeclareLaunchArgument('waypoints_file',
            default_value='',
            description='Ruta absoluta al archivo YAML de waypoints'),
    ]

    debug           = LaunchConfiguration('debug')
    use_gps         = LaunchConfiguration('use_gps')
    use_lidar       = LaunchConfiguration('use_lidar')
    use_stm32       = LaunchConfiguration('use_stm32')
    use_loc         = LaunchConfiguration('use_localization')
    waypoints_file  = LaunchConfiguration('waypoints_file')

    # ── 1. SENSORES ───────────────────────────────────────────────────────────

    camera_node = Node(
        package='av_camera',
        executable='camera_node',
        name='camera_node',
        parameters=[cfg('av_camera', 'camera.yaml')],
        output='screen'
    )

    lidar_node = Node(
        package='av_lidar',
        executable='lidar_node',
        name='lidar_node',
        parameters=[cfg('av_lidar', 'lidar.yaml')],
        output='screen',
        condition=IfCondition(use_lidar)
    )

    gps_node = Node(
        package='av_gps',
        executable='gps_node',
        name='gps_node',
        parameters=[cfg('av_gps', 'gps.yaml')],
        output='screen',
        condition=IfCondition(use_gps)
    )

    stm32_node = Node(
        package='av_stm32',
        executable='stm32_bridge_node',
        name='stm32_bridge_node',
        parameters=[cfg('av_stm32', 'stm32.yaml')],
        output='screen',
        condition=IfCondition(use_stm32)
    )

    # ── 2. PERCEPCIÓN ─────────────────────────────────────────────────────────

    vision_node = Node(
        package='av_vision',
        executable='vision_node',
        name='vision_node',
        parameters=[
            cfg('av_vision', 'vision.yaml'),
            {'debug': debug}
        ],
        output='screen'
    )

    obstacle_node = Node(
        package='av_obstacle',
        executable='obstacle_node',
        name='obstacle_node',
        parameters=[cfg('av_obstacle', 'obstacle.yaml')],
        output='screen',
        condition=IfCondition(use_lidar)
    )

    # ── 3. LOCALIZACIÓN (EKF + navsat_transform) ──────────────────────────────

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg('av_localization'), 'launch',
                         'localization.launch.py')
        ),
        condition=IfCondition(use_loc)
    )

    # ── 4. PLANIFICACIÓN ──────────────────────────────────────────────────────

    planner_node = Node(
        package='av_planner',
        executable='planner_node',
        name='planner_node',
        parameters=[
            cfg('av_planner', 'planner.yaml'),
            {'waypoints_file': waypoints_file}
        ],
        output='screen'
    )

    # ── 5. COMPORTAMIENTO ─────────────────────────────────────────────────────

    behavior_node = Node(
        package='av_behavior',
        executable='behavior_node',
        name='behavior_node',
        parameters=[cfg('av_behavior', 'behavior.yaml')],
        output='screen'
    )

    # ── 6. CONTROL ────────────────────────────────────────────────────────────

    control_node = Node(
        package='av_control',
        executable='control_node',
        name='control_node',
        parameters=[cfg('av_control', 'control.yaml')],
        output='screen'
    )

    return LaunchDescription(args + [
        # Sensores
        camera_node,
        lidar_node,
        gps_node,
        stm32_node,
        # Percepción
        vision_node,
        obstacle_node,
        # Localización
        localization_launch,
        # Planificación y comportamiento
        planner_node,
        behavior_node,
        # Control
        control_node,
    ])