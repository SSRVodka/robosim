#!/usr/bin/env python3
"""Move the Habitat RGB camera through SimulationService.SetObjectPose."""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

from control_stubs.tools.client import RobosimClient

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--camera", default="habitat_rgb")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--look-at", type=float, nargs=3, default=(0.0, 0.8, 0.0))
    parser.add_argument("--yaw", type=float, default=0.0, help="Orbit yaw in degrees")
    parser.add_argument("--pitch", type=float, default=-15.0, help="Orbit pitch in degrees")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--yaw-step", type=float, default=5.0)
    parser.add_argument("--pitch-step", type=float, default=5.0)
    parser.add_argument("--radius-step", type=float, default=0.2)
    parser.add_argument("--show", action="store_true", help="Display the RGB stream after moving")
    parser.add_argument("--save-video", help="Write RGB frames to this video file")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted")
    return parser


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def look_at_quaternion(position: Vector3, target: Vector3) -> Quaternion:
    eye = np.asarray(position, dtype=np.float64)
    center = np.asarray(target, dtype=np.float64)
    forward = normalize(center - eye)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = normalize(np.cross(forward, up_hint))
    up = np.cross(right, forward)

    rotation = np.column_stack((right, up, -forward))
    return matrix_to_quaternion_xyzw(rotation)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> Quaternion:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return float(x), float(y), float(z), float(w)


def orbit_position(target: Vector3, yaw_deg: float, pitch_deg: float, radius: float) -> Vector3:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    target_x, target_y, target_z = target
    x = target_x + radius * math.cos(pitch) * math.sin(yaw)
    y = target_y + radius * math.sin(pitch)
    z = target_z + radius * math.cos(pitch) * math.cos(yaw)
    return x, y, z


def set_camera_pose(
    client: RobosimClient,
    camera: str,
    position: Vector3,
    target: Vector3,
) -> None:
    orientation = look_at_quaternion(position, target)
    status = client.simulation.set_object_pose(camera, position, orientation)
    if status.code and status.code != 1:
        raise RuntimeError(status.message or f"SetObjectPose failed: code={status.code}")


def read_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    data = os.read(sys.stdin.fileno(), 1)
    if not data:
        return None
    return data.decode(errors="ignore")


def run_interactive(args: argparse.Namespace, client: RobosimClient) -> None:
    target = tuple(args.look_at)
    yaw = args.yaw
    pitch = args.pitch
    radius = args.radius
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    print("keys: a/d yaw, w/s pitch, r/f radius, q quit")
    try:
        tty.setcbreak(fd)
        while True:
            position = orbit_position(target, yaw, pitch, radius)
            set_camera_pose(client, args.camera, position, target)
            print(
                f"\rposition=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) "
                f"yaw={yaw:.1f} pitch={pitch:.1f} radius={radius:.2f}",
                end="",
                flush=True,
            )
            key = read_key(0.1)
            if key == "q":
                break
            if key == "a":
                yaw -= args.yaw_step
            elif key == "d":
                yaw += args.yaw_step
            elif key == "w":
                pitch = min(pitch + args.pitch_step, 85.0)
            elif key == "s":
                pitch = max(pitch - args.pitch_step, -85.0)
            elif key == "r":
                radius = max(0.2, radius - args.radius_step)
            elif key == "f":
                radius += args.radius_step
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        print()


def stream_camera(args: argparse.Namespace, client: RobosimClient) -> None:
    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")

    writer = None
    frame_count = 0
    try:
        for data in client.sensing.stream_sensors([args.camera]):
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

            cv2.imshow(args.camera, bgr)
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
            if cv2.waitKey(1) == 27:
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"\nsaved {args.save_video}")
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(args.host, args.port)
    try:
        if args.interactive:
            run_interactive(args, client)
            if args.show or args.save_video:
                stream_camera(args, client)
            return 0
        target = tuple(args.look_at)
        position = tuple(args.position) if args.position else orbit_position(
            target,
            args.yaw,
            args.pitch,
            args.radius,
        )
        set_camera_pose(client, args.camera, position, target)
        print(
            f"set {args.camera} position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
            f"look_at=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
        )
        if args.show or args.save_video:
            stream_camera(args, client)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
