"""Tests for the G1 dual-arm MuJoCo scene."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.mujoco.backend import MuJoCoBackend

G1_DUAL_ARM_SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/unitree_g1/scene.xml"
)


@pytest.fixture
def g1_dual_arm_backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(G1_DUAL_ARM_SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def _wait_for_condition(predicate: Callable[[], bool], timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _joint_position(backend: MuJoCoBackend, joint_name: str) -> float:
    state = backend.get_robot_state()
    return dict(zip(state.name, state.position, strict=True))[joint_name]


def test_g1_dual_arm_spec_exposes_both_arms(g1_dual_arm_backend: MuJoCoBackend) -> None:
    spec = g1_dual_arm_backend.get_robot_spec()

    assert spec.robot_name == "g1_dual_arm"
    groups = {group.name: group for group in spec.joint_model_groups}
    assert set(groups) == {"left_arm", "right_arm", "both_arms"}
    assert list(groups["left_arm"].joint_names) == [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    assert list(groups["right_arm"].joint_names) == [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert [ee.name for ee in groups["left_arm"].end_effectors] == [
        "left_wrist_yaw_link"
    ]
    assert [ee.name for ee in groups["right_arm"].end_effectors] == [
        "right_wrist_yaw_link"
    ]


def test_g1_dual_arm_position_command_moves_joint(
    g1_dual_arm_backend: MuJoCoBackend,
) -> None:
    joint_name = "left_elbow_joint"
    initial_state = g1_dual_arm_backend.get_robot_state()
    initial_positions = dict(zip(initial_state.name, initial_state.position, strict=True))
    initial_position = initial_positions[joint_name]
    target_position = initial_position + 0.25

    g1_dual_arm_backend.set_joint_target(
        names=[joint_name],
        data=[target_position],
        mode=core_pb2.JointCommand.ControlMode.POSITION,
        group="left_arm",
    )

    moved = _wait_for_condition(
        lambda: _joint_position(g1_dual_arm_backend, joint_name) > initial_position + 0.05
    )
    assert moved

    final_state = g1_dual_arm_backend.get_robot_state()
    final_positions = dict(zip(final_state.name, final_state.position, strict=True))
    assert final_positions[joint_name] > initial_position + 0.05
    assert final_positions[joint_name] <= target_position + 0.05


def test_g1_dual_arm_position_command_does_not_flail_neighbor_joints(
    g1_dual_arm_backend: MuJoCoBackend,
) -> None:
    g1_dual_arm_backend.set_joint_target(
        names=["left_elbow_joint"],
        data=[0.25],
        mode=core_pb2.JointCommand.ControlMode.POSITION,
        group="left_arm",
    )

    max_wrist_yaw_abs = 0.0
    max_wrist_yaw_velocity = 0.0
    for _ in range(50):
        time.sleep(0.02)
        state = g1_dual_arm_backend.get_robot_state()
        positions = dict(zip(state.name, state.position, strict=True))
        velocities = dict(zip(state.name, state.velocity, strict=True))
        max_wrist_yaw_abs = max(
            max_wrist_yaw_abs, abs(positions["left_wrist_yaw_joint"])
        )
        max_wrist_yaw_velocity = max(
            max_wrist_yaw_velocity, abs(velocities["left_wrist_yaw_joint"])
        )

    assert max_wrist_yaw_abs < 0.02
    assert max_wrist_yaw_velocity < 0.2
