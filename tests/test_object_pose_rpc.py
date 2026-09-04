from __future__ import annotations

from pathlib import Path

import pytest

from control_stubs import common_pb2
from robosim.backends import MuJoCoBackend

PANDA_SCENE = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim"
    / "mujoco"
    / "assets"
    / "robots"
    / "franka_panda"
    / "scene.xml"
)


@pytest.fixture(scope="module")
def backend() -> MuJoCoBackend:
    instance = MuJoCoBackend(str(PANDA_SCENE), headless=True)
    yield instance
    instance.shutdown()


def test_get_object_pose_roundtrip(backend: MuJoCoBackend) -> None:
    pose = common_pb2.Pose(
        position=common_pb2.Point(x=0.63, y=-0.17, z=0.2),
        orientation=common_pb2.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    backend.set_object_pose("cup", pose)
    read = backend.get_object_pose("cup")
    assert abs(read.position.x - 0.63) < 1e-6
    assert abs(read.position.y + 0.17) < 1e-6
    assert abs(read.position.z - 0.2) < 1e-6
    assert abs(read.orientation.w - 1.0) < 1e-6


def test_get_object_pose_unknown_body(backend: MuJoCoBackend) -> None:
    with pytest.raises(ValueError, match="Unknown MuJoCo body"):
        backend.get_object_pose("nonexistent")
