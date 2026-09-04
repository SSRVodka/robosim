"""ROS Python environment isolation for the active conda environment."""

from __future__ import annotations

import os
import sys

_SYSTEM_ROS_PREFIX = "/opt/ros/"


def flush_ros_environment() -> dict[str, str]:
    """Exclude system ROS Python packages from this process and its children."""
    sys.path[:] = [path for path in sys.path if not path.startswith(_SYSTEM_ROS_PREFIX)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in env.get("PYTHONPATH", "").split(os.pathsep)
        if not path.startswith(_SYSTEM_ROS_PREFIX)
    )
    return env
