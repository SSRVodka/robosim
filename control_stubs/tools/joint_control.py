"""Send one group-scoped position target through the RoboSim gRPC API."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import grpc

from control_stubs import common_pb2
from control_stubs.robot_core_pb2 import JointCommand, JointModelGroupSpec
from control_stubs.tools.client import RobosimClient


def _positions(value: str) -> list[float]:
    try:
        result = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("positions must be comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("positions must not be empty")
    return result


def _group(client: RobosimClient, name: str) -> JointModelGroupSpec:
    for group in client.robot_core.get_robot_spec().joint_model_groups:
        if group.name == name:
            return group
    raise ValueError(f"robot has no joint group '{name}'")


def _wait_for_target(
    client: RobosimClient,
    names: Sequence[str],
    target: Sequence[float],
    *,
    tolerance: float,
    timeout: float,
) -> tuple[dict[str, float], float]:
    deadline = time.monotonic() + timeout
    while True:
        state = client.robot_core.get_robot_state()
        actual = dict(zip(state.name, state.position, strict=True))
        if all(name in actual for name in names):
            error = max(
                abs(actual[name] - value)
                for name, value in zip(names, target, strict=True)
            )
            if error <= tolerance:
                return actual, error
        if time.monotonic() >= deadline:
            raise TimeoutError(f"joint target did not converge within {timeout:g}s")
        time.sleep(0.05)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set one joint group position target via gRPC")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--group", required=True)
    parser.add_argument("--positions", required=True, type=_positions)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.tolerance <= 0 or args.timeout <= 0:
        parser.error("--tolerance and --timeout must be positive")

    client = RobosimClient(args.host, args.port)
    try:
        group = _group(client, args.group)
        names = list(group.joint_names)
        if len(names) != len(args.positions):
            parser.error(
                f"group '{group.name}' requires {len(names)} positions for: {', '.join(names)}"
            )
        status = client.robot_core.set_joint_target(
            names, args.positions, JointCommand.POSITION, group.name
        )
        if status.code != common_pb2.STATUS_SUCCESS:
            print(status.message or "joint command failed", file=sys.stderr)
            return 1
        actual, error = _wait_for_target(
            client,
            names,
            args.positions,
            tolerance=args.tolerance,
            timeout=args.timeout,
        )
        for name, target in zip(names, args.positions, strict=True):
            print(f"{name}: target={target:g} actual={actual[name]:.9g}")
        print(f"converged: max_error={error:.3g}")
        return 0
    except (grpc.RpcError, TimeoutError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
