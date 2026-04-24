"""Tests for the MuJoCo backend."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.mujoco.backend import MuJoCoBackend

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def _wait_for_condition(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_robot_spec_uses_srdf_groups(backend: MuJoCoBackend) -> None:
    spec = backend.get_robot_spec()

    assert spec.robot_name == "panda"
    groups = {group.name: group for group in spec.joint_model_groups}
    assert set(groups) == {"panda_arm", "panda_hand", "panda_arm_hand"}
    assert list(groups["panda_arm"].joint_names) == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    assert [ee.name for ee in groups["panda_arm"].end_effectors] == ["hand"]


def test_backend_starts_from_srdf_ready_state(backend: MuJoCoBackend) -> None:
    state = backend.get_robot_state()
    positions = dict(zip(state.name, state.position, strict=True))

    expected = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
    }
    for joint_name, joint_value in expected.items():
        assert positions[joint_name] == pytest.approx(joint_value, abs=1e-3)


def test_list_and_get_sensors(backend: MuJoCoBackend) -> None:
    sensors = {entry.name: entry.type for entry in backend.list_sensors().entries}

    assert sensors["joint_states"]
    assert sensors["world_camera"]
    assert sensors["ft_sensor_force"]
    assert sensors["ft_sensor_torque"]

    sensor_data = backend.get_sensors(
        ["joint_states", "world_camera", "ft_sensor_force", "ft_sensor_torque"]
    )
    assert len(sensor_data.joints) == 1
    assert sensor_data.joints[0].name == "joint_states"
    assert len(sensor_data.images) == 1
    assert sensor_data.images[0].width == 320
    assert sensor_data.images[0].height == 240
    assert len(sensor_data.forces) == 1
    assert len(sensor_data.torques) == 1


def test_set_joint_target_and_reset_world(backend: MuJoCoBackend) -> None:
    initial = backend.get_robot_state()
    initial_finger = initial.position[7]

    backend.set_joint_target(
        names=["panda_finger_joint1", "panda_finger_joint2"],
        data=[0.03, 0.03],
        mode=core_pb2.JointCommand.ControlMode.POSITION,
        group="panda_hand",
    )

    moved = _wait_for_condition(
        lambda: backend.get_robot_state().position[7] > initial_finger + 1e-3,
        timeout=1.0,
    )
    assert moved

    backend.reset_world(seed=0, randomization_params={})

    reset = _wait_for_condition(
        lambda: abs(backend.get_robot_state().position[7]) < 1e-4,
        timeout=1.0,
    )
    assert reset


def test_idle_loop_holds_initial_configuration(backend: MuJoCoBackend) -> None:
    initial = backend.get_robot_state().position[:7]
    time.sleep(0.3)
    later = backend.get_robot_state().position[:7]

    max_drift = max(
        abs(current - reference)
        for reference, current in zip(initial, later, strict=True)
    )
    assert max_drift < 0.01


def test_servo_control_stream_accepts_twist(backend: MuJoCoBackend) -> None:
    command = core_pb2.ServoCommand(
        twist_cmd=core_pb2.TwistCommand(
            twist=common_pb2.TwistStamped(),
            target_ee=core_pb2.EESpec(name="hand", parent_jmg_name="panda_arm"),
        )
    )
    command.twist_cmd.twist.twist.linear.z = 0.02

    states = backend.servo_control_stream(iter([command]))
    state = next(states)
    assert len(state.name) == 9


def test_get_end_effector_state(backend: MuJoCoBackend) -> None:
    ee_state = backend.get_end_effector_state("panda_arm")

    assert ee_state.pose_stamped.header.frame_id == "world"
    orientation = ee_state.pose_stamped.pose.orientation
    assert any(
        abs(value) > 0.0
        for value in (orientation.x, orientation.y, orientation.z, orientation.w)
    )
