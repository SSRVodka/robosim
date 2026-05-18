"""Backends module."""

from __future__ import annotations

from typing import Any

__all__ = ["GazeboBackend", "HabitatSimBackend", "MuJoCoBackend"]


def __getattr__(name: str) -> Any:
    if name == "GazeboBackend":
        from robosim.backends.gazebo.backend import GazeboBackend

        return GazeboBackend
    if name == "HabitatSimBackend":
        from robosim.backends.habitat.backend import HabitatSimBackend

        return HabitatSimBackend
    if name == "MuJoCoBackend":
        from robosim.backends.mujoco.backend import MuJoCoBackend

        return MuJoCoBackend
    raise AttributeError(name)
