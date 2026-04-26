"""Tests for lidar-aware path follower helpers."""

from __future__ import annotations

import math

import pytest

from control_stubs import sensing_pb2
from robosim.navigation.geometry import Pose2D
from robosim.navigation.path_follower import NavigationParams, _dynamic_obstacle_points


def test_dynamic_obstacle_points_filter_self_hits_and_rear_rays() -> None:
    scan = sensing_pb2.LidarScan(
        name="robot_vacuum_scan",
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        ranges=[0.15, 0.50, 0.50, 2.00],
    )
    params = NavigationParams(
        dynamic_min_distance=0.22,
        dynamic_mark_distance=0.80,
        dynamic_mark_half_angle_deg=100.0,
    )

    points = _dynamic_obstacle_points(scan, Pose2D(1.0, 2.0, 0.0), params)

    assert len(points) == 1
    assert points[0].x == pytest.approx(1.0)
    assert points[0].y == pytest.approx(2.5)
