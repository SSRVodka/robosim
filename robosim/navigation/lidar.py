"""Lidar scan helpers for reactive local navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from control_stubs import sensing_pb2
from control_stubs.tools.client import RobosimClient
from robosim.navigation.geometry import Point2D, Pose2D

DEFAULT_SCAN_NAME = "robot_vacuum_scan"


@dataclass(frozen=True, slots=True)
class LidarSummary:
    front: float
    left_front: float
    right_front: float
    left: float
    right: float
    back: float
    closest_range: float
    closest_angle_deg: float


def angle_diff_deg(angle: float, center: float) -> float:
    return (angle - center + 180.0) % 360.0 - 180.0


def clean_range(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return float("inf")
    return value


def fmt_range(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def iter_scan_angles(scan: sensing_pb2.LidarScan) -> list[tuple[float, float]]:
    angle = float(scan.angle_min)
    step = float(scan.angle_increment)
    return [(angle + step * index, clean_range(value)) for index, value in enumerate(scan.ranges)]


def sector_min(scan: sensing_pb2.LidarScan, center_deg: float, width_deg: float) -> float:
    best = float("inf")
    angle_deg = math.degrees(scan.angle_min)
    step_deg = math.degrees(scan.angle_increment)
    for index, value in enumerate(scan.ranges):
        ray_angle = (angle_deg + step_deg * index) % 360.0
        if abs(angle_diff_deg(ray_angle, center_deg)) <= width_deg / 2.0:
            best = min(best, clean_range(value))
    return best


def closest_ray(scan: sensing_pb2.LidarScan) -> tuple[float, float]:
    best_range = float("inf")
    best_angle_deg = 0.0
    angle_deg = math.degrees(scan.angle_min)
    step_deg = math.degrees(scan.angle_increment)
    for index, value in enumerate(scan.ranges):
        distance = clean_range(value)
        if distance < best_range:
            best_range = distance
            best_angle_deg = (angle_deg + step_deg * index) % 360.0
    return best_range, best_angle_deg


def summarize_scan(scan: sensing_pb2.LidarScan) -> LidarSummary:
    closest_range, closest_angle_deg = closest_ray(scan)
    return LidarSummary(
        front=sector_min(scan, 0.0, 70.0),
        left_front=sector_min(scan, 45.0, 70.0),
        right_front=sector_min(scan, 315.0, 70.0),
        left=sector_min(scan, 90.0, 70.0),
        right=sector_min(scan, 270.0, 70.0),
        back=sector_min(scan, 180.0, 70.0),
        closest_range=closest_range,
        closest_angle_deg=closest_angle_deg,
    )


def get_scan(client: RobosimClient, name: str = DEFAULT_SCAN_NAME) -> sensing_pb2.LidarScan:
    data = client.sensing.get_sensors([name])
    for scan in data.lidars:
        if scan.name == name:
            return scan
    available = ", ".join(scan.name for scan in data.lidars) or "none"
    raise RuntimeError(f"Lidar scan '{name}' not returned. Returned lidars: {available}")


def scan_obstacle_points(
    scan: sensing_pb2.LidarScan,
    pose: Pose2D,
    max_distance: float,
) -> list[Point2D]:
    points: list[Point2D] = []
    for ray_angle, distance in iter_scan_angles(scan):
        if distance > max_distance:
            continue
        world_angle = pose.yaw + ray_angle
        points.append(
            Point2D(
                x=pose.x + distance * math.cos(world_angle),
                y=pose.y + distance * math.sin(world_angle),
            )
        )
    return points
