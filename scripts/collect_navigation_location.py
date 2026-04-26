#!/usr/bin/env python3
"""Collect semantic navigation locations from the current robot pose."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import yaml

from control_stubs.tools.client import RobosimClient
from robosim.navigation.drive import send_base_velocity
from robosim.navigation.geometry import Pose2D, yaw_from_quaternion
from robosim.navigation.locations import DEFAULT_LOCATIONS_FILE, nearest_waypoint

TELEOP_COMMANDS: dict[str, tuple[float, float]] = {
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (0.0, 1.0),
    "d": (0.0, -1.0),
    " ": (0.0, 0.0),
    "x": (0.0, 0.0),
}


def capture_pose(client: RobosimClient) -> tuple[Pose2D, str]:
    pose_stamped = client.mobility.get_robot_pose_in_map()
    pose = pose_stamped.pose
    q = pose.orientation
    return (
        Pose2D(
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        ),
        pose_stamped.header.frame_id or "world",
    )


def load_or_create_config(
    path: Path,
    scene: str | None = None,
    frame: str | None = None,
) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid locations file: {path}")
        config = loaded
    else:
        config = {}

    config.setdefault("scene", scene or path.parent.name)
    config.setdefault("frame", frame or "world")
    config.setdefault("locations", {})
    config.setdefault("waypoints", {})
    config.setdefault("edges", [])
    if scene:
        config["scene"] = scene
    if frame:
        config["frame"] = frame
    if not isinstance(config["locations"], dict):
        raise ValueError("'locations' must be a mapping")
    return config


def rounded_pose_mapping(pose: Pose2D, precision: int) -> dict[str, float]:
    return {
        "x": round(pose.x, precision),
        "y": round(pose.y, precision),
        "yaw": round(pose.yaw, precision),
    }


def make_location_entry(
    pose: Pose2D,
    aliases: list[str],
    precision: int,
    nearest_waypoint_name: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "aliases": aliases,
        "target_pose": rounded_pose_mapping(pose, precision),
    }
    if nearest_waypoint_name:
        entry["nearest_waypoint"] = nearest_waypoint_name
    return entry


def upsert_location(
    config: dict[str, Any],
    name: str,
    entry: dict[str, Any],
    overwrite: bool,
) -> None:
    locations = config["locations"]
    if name in locations and not overwrite:
        raise ValueError(f"Location '{name}' already exists. Use --overwrite to replace it.")
    locations[name] = entry


def resolve_nearest_waypoint(
    config: dict[str, Any],
    pose: Pose2D,
    explicit_name: str | None,
    auto: bool,
) -> str | None:
    if explicit_name:
        return explicit_name
    if not auto:
        return None
    if not config.get("waypoints"):
        return None
    return nearest_waypoint(config, pose)


def save_config(path: Path, config: dict[str, Any], backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        backup_path = path.with_name(f"{path.name}.bak")
        backup_path.write_bytes(path.read_bytes())
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def print_pose(pose: Pose2D, frame: str) -> None:
    print(f"frame={frame} x={pose.x:.3f}, y={pose.y:.3f}, yaw={pose.yaw:.3f}")


def print_teleop_help() -> None:
    print(
        "Teleop commands: "
        "hold w forward, hold s backward, hold a left, hold d right, "
        "space/x stop, Enter capture, q abort."
    )


def print_location_preview(name: str, entry: dict[str, Any]) -> None:
    print(yaml.safe_dump({name: entry}, allow_unicode=True, sort_keys=False).rstrip())


def parse_aliases(values: list[str], interactive_value: str = "") -> list[str]:
    aliases: list[str] = []
    for value in values:
        aliases.extend(item.strip() for item in value.split(",") if item.strip())
    aliases.extend(item.strip() for item in interactive_value.split(",") if item.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        unique.append(alias)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the current robot pose and save it as a semantic navigation "
            "location. Works with any backend that implements GetRobotPoseInMap."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--locations-file", default=str(DEFAULT_LOCATIONS_FILE))
    parser.add_argument("--scene", help="Scene name stored in the locations YAML")
    parser.add_argument("--frame", help="Frame stored in the locations YAML")
    parser.add_argument("--name", help="Location key, for example living_room")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Alias for this location. Can be repeated or comma-separated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak copy before modifying an existing locations file",
    )
    parser.add_argument("--watch", action="store_true", help="Only print current pose repeatedly")
    parser.add_argument("--print-current", action="store_true", help="Only print current pose once")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Before capture, move the robot with hold-to-drive keys in this terminal",
    )
    parser.add_argument("--teleop-linear", type=float, default=0.12)
    parser.add_argument("--teleop-angular", type=float, default=0.8)
    parser.add_argument(
        "--teleop-release-timeout",
        type=float,
        default=0.45,
        help=(
            "Seconds without a repeated movement key before the robot stops. "
            "Increase it if holding a key feels choppy; decrease it if release feels slow."
        ),
    )
    parser.add_argument("--teleop-rate", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--yaw", type=float, help="Override captured yaw")
    parser.add_argument("--x", type=float, help="Manual x, avoids connecting to server")
    parser.add_argument("--y", type=float, help="Manual y, avoids connecting to server")
    parser.add_argument(
        "--nearest-waypoint",
        help="Optional waypoint name for compatibility with older route scripts",
    )
    parser.add_argument(
        "--auto-nearest-waypoint",
        action="store_true",
        help="Fill nearest_waypoint using existing waypoints in the YAML",
    )
    return parser


def manual_pose_from_args(args: argparse.Namespace) -> Pose2D | None:
    if args.x is None and args.y is None:
        return None
    if args.x is None or args.y is None:
        raise ValueError("Use both --x and --y for manual capture")
    return Pose2D(float(args.x), float(args.y), float(args.yaw or 0.0))


def get_pose_for_capture(args: argparse.Namespace) -> tuple[Pose2D, str]:
    manual_pose = manual_pose_from_args(args)
    if manual_pose is not None:
        return manual_pose, args.frame or "world"

    client = RobosimClient(host=args.host, port=args.port)
    try:
        pose, frame = capture_pose(client)
    finally:
        client.close()

    if args.yaw is not None:
        pose = Pose2D(pose.x, pose.y, float(args.yaw))
    return pose, args.frame or frame


def watch_pose(args: argparse.Namespace) -> int:
    client = RobosimClient(host=args.host, port=args.port)
    try:
        while True:
            pose, frame = capture_pose(client)
            print_pose(pose, args.frame or frame)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        client.close()


def teleop_velocity_for_command(
    command: str,
    linear_speed: float,
    angular_speed: float,
) -> tuple[float, float] | None:
    factors = TELEOP_COMMANDS.get(command)
    if factors is None:
        return None
    linear_factor, angular_factor = factors
    return linear_factor * linear_speed, angular_factor * angular_speed


def read_teleop_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    key = sys.stdin.read(1)
    if key == "\x03":
        raise KeyboardInterrupt
    if key in ("\r", "\n"):
        return "\n"
    return key.lower()


def teleop_capture(args: argparse.Namespace) -> tuple[Pose2D, str]:
    if manual_pose_from_args(args) is not None:
        raise ValueError("--teleop cannot be used together with manual --x/--y capture")
    if not sys.stdin.isatty():
        raise ValueError("--teleop requires an interactive terminal")

    client = RobosimClient(host=args.host, port=args.port)
    old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
    last_velocity: tuple[float, float] | None = None
    active_velocity = (0.0, 0.0)
    active_until = 0.0
    status_at = 0.0
    rate_delay = 1.0 / max(1.0, args.teleop_rate)

    try:
        tty.setcbreak(sys.stdin.fileno())
        print_teleop_help()
        print("Move to the desired stopping point, then press Enter to save.")
        while True:
            now = time.monotonic()
            key = read_teleop_key(timeout=rate_delay)
            if key == "\n":
                send_base_velocity(client, 0.0, 0.0)
                print()
                pose, frame = capture_pose(client)
                return pose, args.frame or frame
            if key == "q":
                send_base_velocity(client, 0.0, 0.0)
                print()
                raise ValueError("Capture cancelled")
            if key in ("h", "?"):
                print()
                print_teleop_help()
            if key:
                velocity = teleop_velocity_for_command(
                    key,
                    linear_speed=max(0.0, args.teleop_linear),
                    angular_speed=max(0.0, args.teleop_angular),
                )
                if velocity is not None:
                    active_velocity = velocity
                    active_until = now + max(0.05, args.teleop_release_timeout)

            if active_velocity != (0.0, 0.0) and now > active_until:
                active_velocity = (0.0, 0.0)

            if active_velocity != last_velocity:
                send_base_velocity(client, active_velocity[0], active_velocity[1])
                last_velocity = active_velocity

            if now >= status_at:
                pose, frame = capture_pose(client)
                linear, angular = active_velocity
                print(
                    "\r"
                    f"frame={args.frame or frame} "
                    f"x={pose.x:.3f}, y={pose.y:.3f}, yaw={pose.yaw:.3f} "
                    f"linear={linear:.2f}, angular={angular:.2f} "
                    "Enter=save q=abort",
                    end="",
                    flush=True,
                )
                status_at = now + 0.2
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_settings)
        try:
            send_base_velocity(client, 0.0, 0.0)
        except Exception:
            pass
        client.close()


def interactive_details(args: argparse.Namespace) -> tuple[str, list[str]]:
    name = args.name or input("Location key, e.g. living_room: ").strip()
    if not name:
        raise ValueError("Location key cannot be empty")

    aliases = list(args.alias)
    alias_text = input("Aliases, comma-separated, e.g. 客厅,living room: ").strip()
    return name, parse_aliases(aliases, alias_text)


def main() -> int:
    args = build_parser().parse_args()
    locations_path = Path(args.locations_file)

    if args.watch:
        return watch_pose(args)

    if args.print_current:
        pose, frame = get_pose_for_capture(args)
        print_pose(pose, frame)
        return 0

    try:
        if args.interactive:
            name, aliases = interactive_details(args)
        else:
            if not args.name:
                raise ValueError("Use --name or --interactive")
            name = args.name
            aliases = parse_aliases(args.alias)

        if args.teleop:
            pose, captured_frame = teleop_capture(args)
        else:
            if args.interactive:
                print("Move the robot to the desired safe stopping point, then press Enter.")
                input()
            pose, captured_frame = get_pose_for_capture(args)

        config = load_or_create_config(
            locations_path,
            scene=args.scene,
            frame=args.frame or captured_frame,
        )
        waypoint_name = resolve_nearest_waypoint(
            config,
            pose,
            explicit_name=args.nearest_waypoint,
            auto=args.auto_nearest_waypoint,
        )
        entry = make_location_entry(
            pose,
            aliases=aliases,
            precision=max(0, args.precision),
            nearest_waypoint_name=waypoint_name,
        )

        print("Captured pose:")
        print_pose(pose, args.frame or captured_frame)
        print("Location entry:")
        print_location_preview(name, entry)

        if args.dry_run:
            print("Dry run: not writing file.")
            return 0

        upsert_location(config, name, entry, overwrite=args.overwrite)
        save_config(locations_path, config, backup=not args.no_backup)
        print(f"Saved location '{name}' to {locations_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
