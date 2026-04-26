#!/usr/bin/env python3
"""Drive the robot vacuum to a pose with simple lidar reactive avoidance."""

from __future__ import annotations

import argparse
import math
import time

from test_vacuum_navigate_location import (
    clamp,
    get_xy_yaw,
    normalize_angle,
    send_base_velocity,
    stop_robot,
)

from control_stubs import sensing_pb2
from control_stubs.tools.client import RobosimClient

DEFAULT_SCAN_NAME = "robot_vacuum_scan"


def angle_diff_deg(angle: float, center: float) -> float:
    return (angle - center + 180.0) % 360.0 - 180.0


def clean_range(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return float("inf")
    return value


def sector_min(scan: sensing_pb2.LidarScan, center_deg: float, width_deg: float) -> float:
    best = float("inf")
    angle = math.degrees(scan.angle_min)
    step = math.degrees(scan.angle_increment)

    for index, value in enumerate(scan.ranges):
        ray_angle = (angle + step * index) % 360.0
        if abs(angle_diff_deg(ray_angle, center_deg)) <= width_deg / 2.0:
            best = min(best, clean_range(value))

    return best


def fmt_range(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def get_scan(client: RobosimClient, name: str) -> sensing_pb2.LidarScan:
    data = client.sensing.get_sensors([name])
    for scan in data.lidars:
        if scan.name == name:
            return scan
    available = ", ".join(scan.name for scan in data.lidars) or "none"
    raise RuntimeError(f"Lidar scan '{name}' not returned. Returned lidars: {available}")


def obstacle_bias(left_min: float, right_min: float, side_distance: float) -> float:
    left_pressure = max(0.0, side_distance - left_min)
    right_pressure = max(0.0, side_distance - right_min)
    return right_pressure - left_pressure


def sensor_avoid_go_to_pose(
    client: RobosimClient,
    target_x: float,
    target_y: float,
    scan_name: str,
    timeout: float,
    arrive_tolerance: float,
    stop_distance: float,
    slow_distance: float,
    side_distance: float,
    normal_linear: float,
    slow_linear: float,
    max_angular: float,
    goal_gain: float,
    avoid_gain: float,
    turn_speed: float,
) -> bool:
    start_time = time.monotonic()

    while True:
        if time.monotonic() - start_time > timeout:
            print("Timeout.")
            stop_robot(client)
            return False

        x, y, yaw = get_xy_yaw(client)
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        if distance <= arrive_tolerance:
            print("Arrived.")
            stop_robot(client)
            return True

        scan = get_scan(client, scan_name)
        front_min = sector_min(scan, 0.0, 70.0)
        left_min = sector_min(scan, 45.0, 70.0)
        right_min = sector_min(scan, 315.0, 70.0)

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - yaw)
        goal_angular = goal_gain * heading_error
        avoid_angular = avoid_gain * obstacle_bias(left_min, right_min, side_distance)

        if front_min < stop_distance:
            linear = 0.0
            angular = turn_speed if left_min >= right_min else -turn_speed
            state = "blocked"
        elif front_min < slow_distance:
            linear = slow_linear
            angular = goal_angular + avoid_angular
            state = "slow"
        else:
            linear = min(normal_linear, max(slow_linear, 0.6 * distance))
            angular = goal_angular + avoid_angular
            state = "track"

        if abs(heading_error) > 1.0 and front_min >= stop_distance:
            linear = 0.0

        angular = clamp(angular, -max_angular, max_angular)

        print(
            f"state={state} x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}, "
            f"distance={distance:.3f}, heading_error={heading_error:.3f}, "
            f"front={fmt_range(front_min)}, left={fmt_range(left_min)}, "
            f"right={fmt_range(right_min)}, linear={linear:.3f}, angular={angular:.3f}"
        )

        send_base_velocity(client, linear, angular)
        time.sleep(0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scan-name", default=DEFAULT_SCAN_NAME)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--arrive-tolerance", type=float, default=0.12)
    parser.add_argument("--stop-distance", type=float, default=0.30)
    parser.add_argument("--slow-distance", type=float, default=0.60)
    parser.add_argument("--side-distance", type=float, default=0.35)
    parser.add_argument("--normal-linear", type=float, default=0.18)
    parser.add_argument("--slow-linear", type=float, default=0.06)
    parser.add_argument("--max-angular", type=float, default=1.2)
    parser.add_argument("--goal-gain", type=float, default=1.6)
    parser.add_argument("--avoid-gain", type=float, default=0.9)
    parser.add_argument("--turn-speed", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(host=args.host, port=args.port)

    try:
        ok = sensor_avoid_go_to_pose(
            client=client,
            target_x=args.x,
            target_y=args.y,
            scan_name=args.scan_name,
            timeout=args.timeout,
            arrive_tolerance=args.arrive_tolerance,
            stop_distance=args.stop_distance,
            slow_distance=args.slow_distance,
            side_distance=args.side_distance,
            normal_linear=args.normal_linear,
            slow_linear=args.slow_linear,
            max_angular=args.max_angular,
            goal_gain=args.goal_gain,
            avoid_gain=args.avoid_gain,
            turn_speed=args.turn_speed,
        )
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("Interrupted.")
        stop_robot(client)
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
