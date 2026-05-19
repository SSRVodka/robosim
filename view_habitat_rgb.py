#!/usr/bin/env python3
"""Display or record the Habitat RGB camera stream."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from control_stubs.tools.client import RobosimClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--sensor", default="habitat_rgb")
    parser.add_argument(
        "--save-video",
        help="Write frames to this video file instead of showing a window",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")

    client = RobosimClient(args.host, args.port)
    writer = None
    frame_count = 0
    try:
        for data in client.sensing.stream_sensors([args.sensor]):
            if not data.images:
                continue

            img = data.images[0]
            rgb = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if args.save_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        args.save_video,
                        fourcc,
                        args.fps,
                        (img.width, img.height),
                    )
                writer.write(bgr)
                frame_count += 1
                print(f"wrote frame {frame_count}", end="\r", flush=True)
                if args.max_frames and frame_count >= args.max_frames:
                    break
                time.sleep(1.0 / args.fps)
                continue

            cv2.imshow(args.sensor, bgr)
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
            if cv2.waitKey(1) == 27:
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"\nsaved {args.save_video}")
        client.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
