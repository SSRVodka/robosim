"""CLI tool for LeRobot policy inference via gRPC."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import grpc

from control_stubs import common_pb2, policy_pb2, policy_pb2_grpc


def _print_status(status: common_pb2.Status, success_message: str) -> int:
    if status.code == common_pb2.STATUS_SUCCESS:
        print(success_message)
        return 0
    print(status.message, file=sys.stderr)
    return 1


def _print_policy_status(status: policy_pb2.PolicyStatus) -> int:
    fields = {
        "status": common_pb2.StatusCode.Name(status.status.code),
        "message": status.status.message,
        "loaded": str(status.loaded).lower(),
        "running": str(status.running).lower(),
        "policy_type": status.policy_type,
        "policy_path": status.policy_path,
        "dataset_repo_name": status.dataset_repo_name,
        "device": status.device,
        "jmg_name": status.jmg_name,
        "control_fps": str(status.control_fps),
        "task_text": status.task_text,
        "active_mode": status.active_mode,
    }
    for key, value in fields.items():
        print(f"{key}: {value}")
    return 0 if status.status.code in {
        common_pb2.STATUS_SUCCESS,
        common_pb2.STATUS_RUNNING,
    } else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LeRobot policy inference via gRPC")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    subparsers = parser.add_subparsers(dest="action", required=True)

    load = subparsers.add_parser("load", help="Load a LeRobot policy checkpoint")
    load.add_argument("--policy-path", required=True)
    load.add_argument("--dataset-repo-name", required=True)
    load.add_argument("--device", default="")
    load.add_argument("--task-text", default="")
    load.add_argument("--jmg-name", default="")
    load.add_argument("--control-fps", type=int, default=0)

    start = subparsers.add_parser("start", help="Start the loaded policy")
    start.add_argument("--task-text", default="")
    start.add_argument("--control-fps", type=int, default=0)

    subparsers.add_parser("stop", help="Stop the running policy")
    subparsers.add_parser("status", help="Print policy runtime status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = policy_pb2_grpc.PolicyInferenceServiceStub(channel)

    try:
        if args.action == "load":
            status = stub.LoadPolicy(
                policy_pb2.PolicyLoadRequest(
                    policy_path=args.policy_path,
                    dataset_repo_name=args.dataset_repo_name,
                    device=args.device,
                    task_text=args.task_text,
                    jmg_name=args.jmg_name,
                    control_fps=args.control_fps,
                )
            )
            return _print_status(status, "Policy loaded.")

        if args.action == "start":
            status = stub.StartPolicy(
                policy_pb2.PolicyStartRequest(
                    task_text=args.task_text,
                    control_fps=args.control_fps,
                )
            )
            return _print_status(status, "Policy started.")

        if args.action == "stop":
            return _print_status(stub.StopPolicy(common_pb2.Empty()), "Policy stopped.")

        if args.action == "status":
            return _print_policy_status(stub.GetPolicyStatus(common_pb2.Empty()))
    finally:
        channel.close()

    parser.error(f"unsupported action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
