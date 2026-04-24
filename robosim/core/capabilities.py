"""Capability flags for robot/simulator backends."""

from enum import Flag, auto


class Capability(Flag):
    """Robot/Simulator capability flags."""

    NONE = 0
    JOINT_READ = auto()
    JOINT_WRITE = auto()
    END_EFFECTOR_READ = auto()
    NAVIGATION = auto()
    SENSOR_CAMERA = auto()
    SENSOR_LIDAR = auto()
    SENSOR_IMU = auto()
    SENSOR_JOINT = auto()
    SENSOR_ODOMETRY = auto()
    SENSOR_FORCE_TORQUE = auto()
    SENSOR_ALL = (
        SENSOR_CAMERA
        | SENSOR_LIDAR
        | SENSOR_IMU
        | SENSOR_JOINT
        | SENSOR_ODOMETRY
        | SENSOR_FORCE_TORQUE
    )
    SIMULATION_CONTROL = auto()
    EMERGENCY_STOP = auto()

    SERVO_CAPABLE = JOINT_READ | JOINT_WRITE | END_EFFECTOR_READ
    NAVIGABLE = NAVIGATION | JOINT_READ
