"""CLI tool to signal episode recording start/end via gRPC."""

from __future__ import annotations

import argparse
import sys

import grpc

from control_stubs import common_pb2, robot_data_pb2, robot_data_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal episode recording via gRPC")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start", help="Start recording an episode")
    start.add_argument("--repo-name", required=True, help="Dataset repo name")
    start.add_argument("--task-text", default="", help="Task description")
    start.add_argument("--fps", type=int, default=30, help="Recording fps")

    subparsers.add_parser("end", help="End the current recording")

    args = parser.parse_args()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = robot_data_pb2_grpc.RobotDataServiceStub(channel)

    try:
        if args.action == "start":
            req = robot_data_pb2.RecordOptions(
                repo_name=args.repo_name,
                task_text=args.task_text,
                fps=args.fps,
            )
            resp = stub.RecordEpisodeStart(req)
            if resp.status.code == common_pb2.STATUS_SUCCESS:
                print(f"Recording started. Episode ID: {resp.episode_id}")
                sys.exit(0)
            print(f"Recording failed: {resp.status.message}", file=sys.stderr)
            sys.exit(1)

        if args.action == "end":
            resp = stub.RecordEpisodeEnd(common_pb2.Empty())
            if resp.code == common_pb2.STATUS_SUCCESS:
                print("Recording ended successfully.")
                sys.exit(0)
            print(f"Ending recording failed: {resp.message}", file=sys.stderr)
            sys.exit(1)
    finally:
        channel.close()


if __name__ == "__main__":
    main()
