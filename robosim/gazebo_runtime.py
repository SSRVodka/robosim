"""Run the ROS 2 control processes for one Gazebo CSD realization."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from robosim.core.csd import CsdRealizationManifest
from robosim.core.ros_environment import flush_ros_environment


def _manifest(path: Path) -> CsdRealizationManifest:
    import json

    manifest = CsdRealizationManifest.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    if manifest.backend != "gazebo" or manifest.gazebo_runtime is None:
        raise ValueError("--csd-manifest must be a Gazebo realization with runtime metadata")
    return manifest


def run(manifest_path: Path, *, headless: bool) -> None:
    """Start the runtime process group until interrupted."""
    manifest = _manifest(manifest_path)
    runtime = manifest.gazebo_runtime
    assert runtime is not None
    root = Path(manifest.root_path)
    world = root / runtime.world_file
    urdf = root / runtime.robot_control_urdf
    env = flush_ros_environment()
    env["GAZEBO_RESOURCE_PATH"] = os.pathsep.join(
        filter(None, (str(root), env.get("GAZEBO_RESOURCE_PATH")))
    )
    ros2 = str(Path(sys.prefix) / "bin" / "ros2")
    commands = [
        (
            ros2,
            "run",
            "robot_state_publisher",
            "robot_state_publisher",
            str(urdf),
            "--ros-args",
            "-r",
            f"__ns:={runtime.namespace}",
        ),
        ("gzserver", "--verbose", str(world)),
    ]
    processes = [subprocess.Popen(command, cwd=root, env=env) for command in commands]
    try:
        for controller in ("joint_state_broadcaster", "joint_trajectory_controller"):
            processes.append(subprocess.Popen(
                (
                    ros2,
                    "run",
                    "controller_manager",
                    "spawner",
                    controller,
                    "-c",
                    f"{runtime.namespace}/controller_manager",
                ),
                cwd=root,
                env=env,
            ))
        if not headless:
            processes.append(subprocess.Popen(("gzclient",), cwd=root, env=env))
        while all(process.poll() is None for process in processes[:2]):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Gazebo CSD runtime")
    parser.add_argument("--csd-manifest", required=True, type=Path)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(args.csd_manifest, headless=args.headless)


if __name__ == "__main__":
    main()
