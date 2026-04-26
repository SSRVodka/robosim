#!/usr/bin/env python3
#每隔 0.5 秒读取一次 get_robot_pose_in_map，然后打印 x/y/z/quaternion。
from __future__ import annotations

import argparse
import time

from control_stubs.tools.client import RobosimClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--interval", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(host=args.host, port=args.port)

    try:
        while True:
            pose_stamped = client.mobility.get_robot_pose_in_map()
            pose = pose_stamped.pose
            position = pose.position
            orientation = pose.orientation

            print(
                f"x={position.x:.3f}, "
                f"y={position.y:.3f}, "
                f"z={position.z:.3f}, "
                f"quat=({orientation.x:.3f}, {orientation.y:.3f}, "
                f"{orientation.z:.3f}, {orientation.w:.3f})"
            )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())