#!/usr/bin/env python3
"""Inspect the robot vacuum lidar/rangefinder scan over RoboSim gRPC."""

from __future__ import annotations

import argparse
import math
import time

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


def print_sensor_list(client: RobosimClient) -> None:
    sensors = client.sensing.list_sensors()
    print("Sensors:")
    for item in sensors.entries:
        sensor_type = sensing_pb2.SensorType.Name(item.type)
        print(f"  {item.name}: {sensor_type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scan-name", default=DEFAULT_SCAN_NAME)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--period", type=float, default=0.2)
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(host=args.host, port=args.port)

    try:
        if args.list:
            print_sensor_list(client)

        for _ in range(args.samples):
            scan = get_scan(client, args.scan_name)
            front = sector_min(scan, 0.0, 60.0)
            left = sector_min(scan, 90.0, 70.0)
            right = sector_min(scan, 270.0, 70.0)
            back = sector_min(scan, 180.0, 60.0)

            print(
                f"scan={scan.name} rays={len(scan.ranges)} "
                f"front={fmt_range(front)} left={fmt_range(left)} "
                f"right={fmt_range(right)} back={fmt_range(back)}"
            )
            time.sleep(args.period)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
