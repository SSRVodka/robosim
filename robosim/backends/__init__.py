"""Backends module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robosim.backends.gazebo.backend import GazeboBackend
    from robosim.backends.mujoco.backend import MuJoCoBackend
    from robosim.backends.pybullet.backend import PyBulletBackend

__all__ = ["GazeboBackend", "MuJoCoBackend", "PyBulletBackend"]


def __getattr__(name: str) -> object:
    if name == "GazeboBackend":
        from robosim.backends.gazebo.backend import GazeboBackend

        return GazeboBackend
    if name == "MuJoCoBackend":
        from robosim.backends.mujoco.backend import MuJoCoBackend

        return MuJoCoBackend
    if name == "PyBulletBackend":
        from robosim.backends.pybullet.backend import PyBulletBackend

        return PyBulletBackend
    raise AttributeError(name)
