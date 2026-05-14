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
            default_value='true',
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
        DeclareLaunchArgument('use_camera_stream',
            default_value='true',
            description='Habilita web_video_server para stream HMI'),
        DeclareLaunchArgument('stream_port',
            default_value='8081',
            description='Puerto HTTP del web_video_server'),
        DeclareLaunchArgument('waypoints_file',
            default_value='',
            description='Ruta al archivo YAML de waypoints GPS'),
        DeclareLaunchArgument('model_path',
            default_value='/home/carrito/best.pt',
            description='Ruta al modelo YOLOv8 de señales'),
    ]

    debug           = LaunchConfiguration('debug')
    use_gps         = LaunchConfiguration('use_gps')
    use_lidar       = LaunchConfiguration('use_lidar')
    use_loc         = LaunchConfiguration('use_localization')
    use_hc12        = LaunchConfiguration('use_hc12')
    use_sign        = LaunchConfiguration('use_traffic_sign')
    use_stream      = LaunchConfiguration('use_camera_stream')
    stream_port     = LaunchConfiguration('stream_port')
    waypoints_file  = LaunchConfiguration('waypoints_file')
    model_path      = LaunchConfiguration('model_path')

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. HARDWARE — STM32 + micro-ROS agent
    # ═══════════════════════════════════════════════════════════════════════════

    stm32_bridge = Node(
        package='av_stm32',
        executable='stm32_bridge',
        name='stm32_bridge',
        parameters=[{
            'port': '/dev/ttyAMA0',
            'baud': 115200,
        }],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. COMUNICACIÓN — HC-12 bridge (control manual via radio)
    # ═══════════════════════════════════════════════════════════════════════════

    hc12_node = Node(
        package='av_stm32',
        executable='hc12_bridge',
        name='hc12_bridge',
        parameters=[{
            'port': '/dev/ttyHC12',
            'baud': 115200,
        }],
        output='screen',
        condition=IfCondition(use_hc12),
        respawn=True,
        respawn_delay=2.0,
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

    # Stream de cámara para el HMI — se lanza junto con la cámara
    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        parameters=[{
            'port': stream_port,
            'address': '0.0.0.0',
            'default_stream_type': 'mjpeg',
        }],
        output='screen',
        condition=IfCondition(use_stream),
    )
    video_recorder = Node(
        package='av_vision',
        executable='video_recorder_node',
        name='video_recorder',
        parameters=[{
            'output_dir': '/home/carrito/videos',
            'fps': 15.0,
            'width': 640,
            'height': 480,
        }],
        output='screen',
    )

    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[cfg('av_lidar', 'lidar.yaml')],
        output='screen',
        condition=IfCondition(use_lidar),
    )

    av_lidar_node = Node(
        package='av_lidar',
        executable='lidar_node',
        name='lidar_node',
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
        prefix = 'taskset -c 3',
        parameters=[{
            'model_path':      model_path,
            'conf_threshold':  0.5,
            'device':          'cpu',
            'imgsz':           160,
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
        respawn=True,
        respawn_delay=2.0,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Orden de lanzamiento con delays para evitar race conditions
    # ═══════════════════════════════════════════════════════════════════════════
    return LaunchDescription(args + [

        # 1. Primero el agente — debe estar listo antes que el STM32
        stm32_bridge,

        # 2. 2s — STM32, HC-12, sensores pasivos y stream de cámara
        TimerAction(period=2.0, actions=[
            hc12_node,
            camera_node,
            web_video_server_node,  # se lanza junto con la cámara
            gps_node,
            video_recorder,
        ]),

        # 3. 4s — LiDAR necesita hardware listo
        TimerAction(period=4.0, actions=[
            lidar_node,
            av_lidar_node,
        ]),

        # 4. 6s — percepción necesita cámara y LiDAR listos
        TimerAction(period=6.0, actions=[
            vision_node,
            traffic_sign_node,
            obstacle_node,
        ]),

        # 5. 8s — localización necesita IMU y GPS publicando
        TimerAction(period=8.0, actions=[
            localization_node,
        ]),

        # 6. 10s — planificación y control al final
        TimerAction(period=10.0, actions=[
            planner_node,
            behavior_node,
            control_node,
        ]),
    ])