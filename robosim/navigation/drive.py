"""Differential-drive helpers for the robot vacuum base."""

from __future__ import annotations

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient
from robosim.navigation.geometry import Pose2D, yaw_from_quaternion

LEFT_JOINT = "rv_left_wheel_joint"
RIGHT_JOINT = "rv_right_wheel_joint"
BASE_GROUP = "base_wheels"

WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.32


def get_xy_yaw(client: RobosimClient) -> Pose2D:
    pose_stamped = client.mobility.get_robot_pose_in_map()
    pose = pose_stamped.pose
    q = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
    )


def wheel_speeds(linear: float, angular: float) -> tuple[float, float]:
    left_linear = linear - angular * WHEEL_BASE / 2.0
    right_linear = linear + angular * WHEEL_BASE / 2.0
    return left_linear / WHEEL_RADIUS, right_linear / WHEEL_RADIUS


def send_base_velocity(client: RobosimClient, linear: float, angular: float) -> None:
    left, right = wheel_speeds(linear, angular)
    client.robot_core.set_joint_target(
        names=[LEFT_JOINT, RIGHT_JOINT],
        data=[left, right],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        jmg_name=BASE_GROUP,
    )


def stop_robot(client: RobosimClient) -> None:
    send_base_velocity(client, 0.0, 0.0)
    client.robot_core.emergency_stop()
