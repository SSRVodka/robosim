#!/usr/bin/env python3
"""Interactively test robot vacuum wheel velocity commands via RoboSim gRPC.

Run this from the repository root after starting the MuJoCo RoboSim server.
The script sends a few low-speed left/right wheel velocity combinations so you
can observe which signs correspond to forward, backward, and turning motion.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import grpc

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient


@dataclass(frozen=True)
class WheelTest:
    label: str
    left: float
    right: float


def _stop_robot(client: RobosimClient, left_joint: str, right_joint: str, group: str) -> None:
    try:
        client.robot_core.set_joint_target(
            names=[left_joint, right_joint],
            data=[0.0, 0.0],
            mode=core_pb2.JointCommand.ControlMode.VELOCITY,
            jmg_name=group,
        )
    finally:
        client.robot_core.emergency_stop()


def _print_spec(client: RobosimClient) -> None:
    spec = client.robot_core.get_robot_spec()
    print(f"Connected robot: {spec.robot_name}")
    print("Joint groups:")
    for group in spec.joint_model_groups:
        print(f"  - {group.name}: {list(group.joint_names)}")


def _run_one_test(
    client: RobosimClient,
    test: WheelTest,
    *,
    left_joint: str,
    right_joint: str,
    group: str,
    duration: float,
    reset_between: bool,
) -> None:
    print()
    print("=" * 72)
    print(f"Test: {test.label}")
    print(f"Command: {left_joint}={test.left:.4f}, {right_joint}={test.right:.4f}")
    print("Watch the MuJoCo window and observe: forward/backward/turn/tip/no motion.")
    input("Press Enter to send this command...")

    if reset_between:
        print("Resetting world...")
        client.simulation.reset_world(seed=0, randomization_params={})
        time.sleep(0.5)

    print(f"Sending command for {duration:.1f}s...")
    client.robot_core.set_joint_target(
        names=[left_joint, right_joint],
        data=[test.left, test.right],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        jmg_name=group,
    )
    time.sleep(duration)
    _stop_robot(client, left_joint, right_joint, group)
    print("Stopped.")
    input("Write down what happened, then press Enter for the next test...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="RoboSim gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="RoboSim gRPC port")
    parser.add_argument("--group", default="base_wheels", help="Joint model group name")
    parser.add_argument("--left-joint", default="rv_left_wheel_joint", help="Left wheel joint name")
    parser.add_argument("--right-joint", default="rv_right_wheel_joint", help="Right wheel joint name")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="Absolute wheel velocity target used in the sign tests",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Seconds to hold each wheel command",
    )
    parser.add_argument(
        "--no-reset-between",
        action="store_true",
        help="Do not reset the world before each test",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    speed = abs(args.speed)
    tests = [
        WheelTest("same positive", speed, speed),
        WheelTest("same negative", -speed, -speed),
        WheelTest("left positive, right negative", speed, -speed),
        WheelTest("left negative, right positive", -speed, speed),
    ]

    client = RobosimClient(host=args.host, port=args.port)
    try:
        _print_spec(client)
        print()
        print("This script will run four low-speed wheel sign tests.")
        print("If all tests only make the robot tip, rerun with an even smaller --speed.")
        print("Example: python3 scripts/test_vacuum_wheels.py --speed 0.01")

        for test in tests:
            _run_one_test(
                client,
                test,
                left_joint=args.left_joint,
                right_joint=args.right_joint,
                group=args.group,
                duration=args.duration,
                reset_between=not args.no_reset_between,
            )
        print()
        print("Done. The useful result is the table of what each sign combination did.")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.code().name}: {exc.details()}")
        return 1
    finally:
        try:
            _stop_robot(client, args.left_joint, args.right_joint, args.group)
        except Exception:
            pass
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
