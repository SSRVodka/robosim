"""Open a backend-native scene in its local viewer without starting gRPC."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Launch the requested backend viewer until its window closes."""
    parser = argparse.ArgumentParser(description="View one backend-native scene.")
    parser.add_argument("--backend", required=True, choices=("mujoco", "pybullet", "gazebo"))
    parser.add_argument(
        "--entry", required=True, type=Path, help="MJCF, PyBullet scene.py, or SDF world"
    )
    args = parser.parse_args()
    entry = args.entry.resolve()
    if not entry.is_file():
        parser.error(f"entry file does not exist: {entry}")
    if args.backend == "mujoco":
        _view_mujoco(entry)
    elif args.backend == "pybullet":
        _view_pybullet(entry)
    else:
        _view_gazebo(entry)


def _view_mujoco(entry: Path) -> None:
    import mujoco.viewer

    mujoco.viewer.launch_from_path(str(entry))


def _view_pybullet(entry: Path) -> None:
    import pybullet as pybullet

    from robosim.backends.pybullet import PyBulletBackend

    backend = PyBulletBackend(scene_path=str(entry), headless=False)
    try:
        while pybullet.isConnected():
            time.sleep(0.1)
    finally:
        backend.shutdown()


def _view_gazebo(entry: Path) -> None:
    subprocess.run(("gazebo", str(entry)), check=True)


if __name__ == "__main__":
    main()
