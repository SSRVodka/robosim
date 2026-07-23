from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control_stubs.tools.scripted_collect import (
    BOX_HOME,
    GRASP_QUAT_WXYZ,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_Q,
    PandaKinematics,
    build_episode_targets,
)

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
def kinematics() -> PandaKinematics:
    return PandaKinematics(PANDA_SCENE)


def test_ik_converges_to_reachable_targets(kinematics: PandaKinematics) -> None:
    for target in ([0.6, -0.2, 0.45], [0.6, -0.2, 0.30], [0.4, 0.3, 0.50]):
        q = kinematics.solve(np.array(target), GRASP_QUAT_WXYZ, HOME_Q)
        assert np.linalg.norm(kinematics.hand_position(q) - target) < 1e-3


def test_ik_raises_for_unreachable_target(kinematics: PandaKinematics) -> None:
    with pytest.raises(ValueError, match="IK did not converge"):
        kinematics.solve(np.array([2.0, 0.0, 0.5]), GRASP_QUAT_WXYZ, HOME_Q)


def test_episode_targets_are_finite_and_bounded(kinematics: PandaKinematics) -> None:
    targets = build_episode_targets(kinematics, BOX_HOME.copy())

    assert 50 <= len(targets) <= 500
    grippers = {gripper for _, gripper in targets}
    assert grippers == {GRIPPER_OPEN, GRIPPER_CLOSED}
    for arm_q, _ in targets:
        assert arm_q.shape == (7,)
        assert np.all(np.isfinite(arm_q))
    steps = np.diff(np.stack([arm_q for arm_q, _ in targets]), axis=0)
    assert np.abs(steps).max() < 0.5


def test_episode_targets_end_open_above_container(kinematics: PandaKinematics) -> None:
    targets = build_episode_targets(kinematics, BOX_HOME.copy())
    final_q, final_gripper = targets[-1]

    assert final_gripper == GRIPPER_OPEN
    hand = kinematics.hand_position(final_q)
    assert np.linalg.norm(hand[:2] - [0.4, 0.3]) < 5e-3
