"""Scene articulation must not be absorbed into the robot joint set.

Provider-supplied scenes contain articulated furniture (drawers, doors) whose
hinge/slide joints belong to the scene, not to the robot. Treating them as
robot joints corrupts the recorded state vector and holds them shut with PD
torque, which would break any open-drawer/open-door task.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from robosim.backends.mujoco.backend import MuJoCoBackend

ARTICULATED_SCENE = (
    Path(__file__).resolve().parent / "fixtures" / "articulated_scene" / "scene.xml"
)
PANDA_JOINTS = [f"panda_joint{index}" for index in range(1, 8)] + [
    "panda_finger_joint1",
    "panda_finger_joint2",
]
SCENE_JOINTS = ["cabinet/drawer_joint", "cabinet/door_joint"]


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(ARTICULATED_SCENE), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def test_robot_state_excludes_scene_articulation(backend: MuJoCoBackend) -> None:
    names = list(backend.get_robot_state().name)
    assert names == PANDA_JOINTS
    assert not [name for name in names if name in SCENE_JOINTS]


def test_scene_articulation_is_not_held_by_controllers(
    backend: MuJoCoBackend,
) -> None:
    # 场景铰接自由度必须保持无驱动：被当作机器人关节时会收到 PD 保持力矩而焊死。
    model = backend._model  # noqa: SLF001 - white-box check of the hold set
    data = backend._data  # noqa: SLF001
    for joint_name in SCENE_JOINTS:
        dof_adr = int(model.jnt_dofadr[model.joint(joint_name).id])
        assert data.qfrc_applied[dof_adr] == 0.0


def test_scene_articulation_stays_free_to_move(backend: MuJoCoBackend) -> None:
    model = backend._model  # noqa: SLF001
    data = backend._data  # noqa: SLF001
    drawer = model.joint("cabinet/drawer_joint")
    qpos_adr = int(model.jnt_qposadr[drawer.id])
    with backend._state_lock:  # noqa: SLF001
        data.qpos[qpos_adr] = -0.1
    import time

    time.sleep(0.5)
    assert data.qpos[qpos_adr] == pytest.approx(-0.1, abs=2e-2)


def test_robot_spec_groups_cover_only_robot_joints(backend: MuJoCoBackend) -> None:
    spec = backend.get_robot_spec()
    spec_joint_names = {joint.name for joint in spec.joints}
    assert spec_joint_names == set(PANDA_JOINTS)
