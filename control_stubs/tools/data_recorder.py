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
    subparsers.add_parser("cancel", help="Discard the current recording")
    replay = subparsers.add_parser("replay", help="Replay a recorded episode")
    replay.add_argument("--repo-name", required=True, help="Dataset repo name")
    replay.add_argument("--episode-id", required=True, type=int, help="Episode ID to replay")

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
            resp = stub.EpisodeStart(req)
            if resp.status.code == common_pb2.STATUS_SUCCESS:
                print(f"Recording started. Episode ID: {resp.episode_id}")
                sys.exit(0)
            print(f"Recording failed: {resp.status.message}", file=sys.stderr)
            sys.exit(1)

        if args.action == "end":
            resp = stub.EpisodeEnd(common_pb2.Empty())
            if resp.code == common_pb2.STATUS_SUCCESS:
                print("Recording ended successfully.")
                sys.exit(0)
            print(f"Ending recording failed: {resp.message}", file=sys.stderr)
            sys.exit(1)

        if args.action == "cancel":
            resp = stub.EpisodeCancel(common_pb2.Empty())
            if resp.code == common_pb2.STATUS_SUCCESS:
                print("Recording cancelled successfully.")
                sys.exit(0)
            print(f"Cancelling recording failed: {resp.message}", file=sys.stderr)
            sys.exit(1)

        if args.action == "replay":
            req = robot_data_pb2.RecordInfo(
                repo_name=args.repo_name,
                episode_id=args.episode_id,
            )
            resp = stub.EpisodeReplay(req)
            if resp.code == common_pb2.STATUS_SUCCESS:
                print("Replay finished successfully.")
                sys.exit(0)
            print(f"Replay failed: {resp.message}", file=sys.stderr)
            sys.exit(1)
    finally:
        channel.close()


if __name__ == "__main__":
    main()
