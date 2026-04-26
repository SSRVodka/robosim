#!/usr/bin/env python3
"""Exercise MobilityService.NavigateTo for the robot vacuum navigation stack."""

from __future__ import annotations

import argparse
import math

from control_stubs import common_pb2
from control_stubs.tools.client import RobosimClient
from robosim.navigation.geometry import Pose2D
from robosim.navigation.locations import (
    DEFAULT_LOCATIONS_FILE,
    load_locations,
    target_pose_for_location,
)


def resolve_target(args: argparse.Namespace) -> Pose2D:
    if args.location:
        config = load_locations(args.locations_file)
        return target_pose_for_location(config, args.location)
    if args.x is None or args.y is None:
        raise ValueError("Use either --location or both --x and --y")
    return Pose2D(float(args.x), float(args.y), float(args.yaw))


def yaw_to_quaternion_z_w(yaw: float) -> tuple[float, float]:
    return math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS_FILE))
    parser.add_argument("--location")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--target-frame", default="world")
    parser.add_argument("--max-velocity", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target = resolve_target(args)
    qz, qw = yaw_to_quaternion_z_w(target.yaw)

    client = RobosimClient(host=args.host, port=args.port)
    last_status = common_pb2.STATUS_UNKNOWN
    try:
        feedback_stream = client.mobility.navigate_to(
            target_pose=(target.x, target.y, 0.0, 0.0, 0.0, qz, qw),
            target_frame=args.target_frame,
            max_velocity=args.max_velocity,
        )
        for feedback in feedback_stream:
            last_status = feedback.status.code
            print(
                f"status={common_pb2.StatusCode.Name(feedback.status.code)} "
                f"message={feedback.status.message} "
                f"eta={feedback.eta} "
                f"text={feedback.feedback_text}"
            )
    finally:
        client.close()

    return 0 if last_status == common_pb2.STATUS_SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
