# MERAKI — Autonomous Navigation System

> 1:4 scale autonomous vehicle built as a mechatronics thesis project at 
> Universidad Tecnológica de Bolívar, Colombia.

MERAKI integrates a multimodal perception stack (LiDAR, RGB camera, IMU, 
GPS, ultrasonic) with a full ROS 2 Jazzy architecture running on a 
Raspberry Pi 5. Low-level control runs bare-metal on an STM32F411, 
communicating over UART/JSON. A web-based HMI provides real-time 
telemetry, LiDAR visualization and GPS route planning over XBee.

## Key results
- Lane following: detection quality 1.0, lateral offset 0.022 m  
- Traffic sign detection: mean confidence 0.78 (YOLOv8n, 10 Colombian classes)  
- Energy autonomy: 1 h 42 min continuous operation  
- Top speed: 10 km/h at 30% PWM duty cycle  

## Stack
ROS 2 Jazzy · Python · C (STM32 HAL) · OpenCV · YOLOv8n · 
Raspberry Pi 5 · STM32F411 · RPLIDAR S2L · XBee Pro S1
