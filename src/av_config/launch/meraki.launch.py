#!/usr/bin/env python3
"""
MERAKI — Launch principal
Lanza todo el sistema de navegación autónoma en la RPi5

Uso:
  ros2 launch av_config meraki.launch.py
  ros2 launch av_config meraki.launch.py debug:=true
  ros2 launch av_config meraki.launch.py use_gps:=false use_lidar:=false
  ros2 launch av_config meraki.launch.py waypoints_file:=/home/carrito/waypoints.yaml
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def pkg(name: str) -> str:
    return get_package_share_directory(name)


def cfg(package: str, yaml: str) -> str:
    return os.path.join(pkg(package), 'config', yaml)


def generate_launch_description():

    # ── Argumentos ────────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('debug',
            default_value='false',
            description='Publica topics de debug'),
        DeclareLaunchArgument('use_gps',
            default_value='true',
            description='Habilita GPS NEO-8M'),
        DeclareLaunchArgument('use_lidar',
            default_value='true',
            description='Habilita LiDAR RPLIDAR S2'),
        DeclareLaunchArgument('use_localization',
            default_value='true',
            description='Habilita stack de localización'),
        DeclareLaunchArgument('use_hc12',
            default_value='true',
            description='Habilita bridge HC-12 radio'),
        DeclareLaunchArgument('use_traffic_sign',
            default_value='true',
            description='Habilita detección de señales YOLOv8'),
        DeclareLaunchArgument('waypoints_file',
            default_value='',
            description='Ruta al archivo YAML de waypoints GPS'),
        DeclareLaunchArgument('model_path',
            default_value='/home/carrito/models/best.pt',
            description='Ruta al modelo YOLOv8 de señales'),
    ]

    debug           = LaunchConfiguration('debug')
    use_gps         = LaunchConfiguration('use_gps')
    use_lidar       = LaunchConfiguration('use_lidar')
    use_loc         = LaunchConfiguration('use_localization')
    use_hc12        = LaunchConfiguration('use_hc12')
    use_sign        = LaunchConfiguration('use_traffic_sign')
    waypoints_file  = LaunchConfiguration('waypoints_file')
    model_path      = LaunchConfiguration('model_path')

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. HARDWARE — STM32 + micro-ROS agent
    # ═══════════════════════════════════════════════════════════════════════════

    stm32_node = Node(
        package='av_stm32',
        executable='stm32_node',
        name='stm32_node',
        output='screen',
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. COMUNICACIÓN — HC-12 bridge (control manual via radio)
    # ═══════════════════════════════════════════════════════════════════════════

    hc12_node = Node(
        package='av_stm32',
        executable='hc12_bridge',
        name='hc12_bridge',
        parameters=[{
            'port': '/dev/ttyUSB1',
            'baud': 9600,
        }],
        output='screen',
        condition=IfCondition(use_hc12),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. SENSORES
    # ═══════════════════════════════════════════════════════════════════════════

    camera_node = Node(
        package='av_camera',
        executable='camera_node',
        name='camera_node',
        parameters=[cfg('av_camera', 'camera.yaml')],
        output='screen',
    )

    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='rplidar_node',
        parameters=[cfg('av_lidar', 'lidar.yaml')],
        output='screen',
        condition=IfCondition(use_lidar),
    )

    gps_node = Node(
        package='av_gps',
        executable='gps_node',
        name='gps_node',
        parameters=[cfg('av_gps', 'gps.yaml')],
        output='screen',
        condition=IfCondition(use_gps),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. PERCEPCIÓN
    # ═══════════════════════════════════════════════════════════════════════════

    vision_node = Node(
        package='av_vision',
        executable='vision_node',
        name='vision_node',
        parameters=[
            cfg('av_vision', 'vision.yaml'),
            {'debug': debug},
        ],
        output='screen',
    )

    traffic_sign_node = Node(
        package='av_vision',
        executable='traffic_sign_node',
        name='traffic_sign_node',
        parameters=[{
            'model_path':      model_path,
            'conf_threshold':  0.5,
            'device':          'cpu',
            'imgsz':           320,
            'detect_vehicles': True,
            'debug':           debug,
        }],
        output='screen',
        condition=IfCondition(use_sign),
    )

    obstacle_node = Node(
        package='av_obstacle',
        executable='obstacle_node',
        name='obstacle_node',
        parameters=[cfg('av_obstacle', 'obstacle.yaml')],
        output='screen',
        condition=IfCondition(use_lidar),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. LOCALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════════

    localization_node = Node(
        package='av_localization',
        executable='localization_node',
        name='localization_node',
        parameters=[cfg('av_localization', 'localization.yaml')],
        output='screen',
        condition=IfCondition(use_loc),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. PLANIFICACIÓN
    # ═══════════════════════════════════════════════════════════════════════════

    planner_node = Node(
        package='av_planner',
        executable='planner_node',
        name='planner_node',
        parameters=[
            cfg('av_planner', 'planner.yaml'),
            {'waypoints_file': waypoints_file},
        ],
        output='screen',
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. COMPORTAMIENTO Y CONTROL
    # ═══════════════════════════════════════════════════════════════════════════

    behavior_node = Node(
        package='av_behavior',
        executable='behavior_node',
        name='behavior_node',
        parameters=[cfg('av_behavior', 'behavior.yaml')],
        output='screen',
    )

    control_node = Node(
        package='av_control',
        executable='control_node',
        name='control_node',
        parameters=[cfg('av_control', 'control.yaml')],
        output='screen',
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Orden de lanzamiento con delays para evitar race conditions
    # ═══════════════════════════════════════════════════════════════════════════
    return LaunchDescription(args + [
        # Inmediato — hardware y comunicación
        stm32_node,
        hc12_node,
        camera_node,
        gps_node,

        # 2s delay — sensores que necesitan hardware listo
        TimerAction(period=2.0, actions=[lidar_node]),

        # 3s delay — percepción necesita cámara lista
        TimerAction(period=3.0, actions=[
            vision_node,
            traffic_sign_node,
            obstacle_node,
        ]),

        # 4s delay — localización necesita IMU y GPS
        TimerAction(period=4.0, actions=[localization_node]),

        # 5s delay — planificación y comportamiento al final
        TimerAction(period=5.0, actions=[
            planner_node,
            behavior_node,
            control_node,
        ]),
    ])