"""Tests for the MuJoCo backend."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.mujoco.backend import MuJoCoBackend

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)
OLDROOM_SCENE_PATH = (
    Path(__file__).resolve().parent.parent / "drivers_sim/mujoco/assets/worlds/oldroom/scene.xml"
)
BEDROOM_SCENE_PATH = (
    Path(__file__).resolve().parent.parent / "drivers_sim/mujoco/assets/worlds/bedroom/scene.xml"
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


def test_camera_rendering_remains_valid_across_threads(backend: MuJoCoBackend) -> None:
    first_image = backend.get_sensors(["world_camera"]).images[0]

    with ThreadPoolExecutor(max_workers=1) as executor:
        second_image = executor.submit(
            lambda: backend.get_sensors(["world_camera"]).images[0]
        ).result()

    first_frame = np.frombuffer(first_image.data, dtype=np.uint8).reshape(
        first_image.height,
        first_image.width,
        3,
    )
    second_frame = np.frombuffer(second_image.data, dtype=np.uint8).reshape(
        second_image.height,
        second_image.width,
        3,
    )

    assert float(first_frame.mean()) > 20.0
    assert float(first_frame.std()) > 20.0
    assert float(second_frame.mean()) > 20.0
    assert float(second_frame.std()) > 20.0
    assert np.mean(np.abs(second_frame.astype(np.int16) - first_frame.astype(np.int16))) < 10.0


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


def test_joint_command_state_stays_replayable_during_velocity_control(
    backend: MuJoCoBackend,
) -> None:
    initial_command_state = backend.get_joint_command_state()
    initial_position_map = dict(
        zip(initial_command_state.name, initial_command_state.position, strict=True)
    )
    backend.set_joint_target(
        names=["panda_joint2"],
        data=[0.2],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        group="panda_arm",
    )
    time.sleep(0.05)

    robot_state = backend.get_robot_state()
    command_state = backend.get_joint_command_state()
    position_map = dict(zip(command_state.name, command_state.position, strict=True))
    velocity_map = dict(zip(command_state.name, command_state.velocity, strict=True))
    robot_position_map = dict(zip(robot_state.name, robot_state.position, strict=True))

    assert position_map["panda_joint2"] == pytest.approx(
        robot_position_map["panda_joint2"],
        abs=1e-3,
    )
    assert velocity_map["panda_joint2"] == pytest.approx(0.2)
    assert position_map["panda_joint6"] == pytest.approx(
        initial_position_map["panda_joint6"],
        abs=1e-6,
    )


def test_idle_loop_holds_initial_configuration(backend: MuJoCoBackend) -> None:
    initial = backend.get_robot_state().position[:7]
    time.sleep(0.3)
    later = backend.get_robot_state().position[:7]

    max_drift = max(
        abs(current - reference) for reference, current in zip(initial, later, strict=True)
    )
    assert max_drift < 0.01


def test_oldroom_vx300s_home_starts_stable() -> None:
    backend = MuJoCoBackend(str(OLDROOM_SCENE_PATH), headless=True)
    try:
        time.sleep(0.3)
        with backend._state_lock:
            data = backend._data
            model = backend._model
            arm_qpos = data.qpos[13:29].copy()
            max_qacc = float(np.max(np.abs(data.qacc)))
            arm_contacts = []
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                geom1 = model.geom(contact.geom1).name
                geom2 = model.geom(contact.geom2).name
                if "vx300s" in geom1 or "vx300s" in geom2:
                    arm_contacts.append((geom1, geom2))

        expected = np.array(
            [
                0.0,
                -0.96,
                1.16,
                0.0,
                -0.3,
                0.0,
                0.03,
                -0.03,
                0.0,
                -0.96,
                1.16,
                0.0,
                -0.3,
                0.0,
                0.03,
                -0.03,
            ]
        )
        assert arm_qpos == pytest.approx(expected, abs=1e-4)
        assert max_qacc < 1e-3
        assert arm_contacts == []
    finally:
        backend.shutdown()


def test_oldroom_position_actuated_arm_velocity_hold_is_stable() -> None:
    backend = MuJoCoBackend(str(OLDROOM_SCENE_PATH), headless=True)
    arm_names = [
        "vx300s_left/waist",
        "vx300s_left/shoulder",
        "vx300s_left/elbow",
        "vx300s_left/forearm_roll",
        "vx300s_left/wrist_angle",
        "vx300s_left/wrist_rotate",
    ]
    try:
        initial = backend.get_robot_state()
        initial_positions = dict(zip(initial.name, initial.position, strict=True))

        backend.set_joint_target(
            names=arm_names,
            data=[0.0] * len(arm_names),
            mode=core_pb2.JointCommand.ControlMode.VELOCITY,
            group="left_arm",
        )
        time.sleep(0.3)

        state = backend.get_robot_state()
        positions = dict(zip(state.name, state.position, strict=True))
        with backend._state_lock:
            max_qacc = float(np.max(np.abs(backend._data.qacc)))

        for name in arm_names:
            assert positions[name] == pytest.approx(initial_positions[name], abs=1e-3)
        assert np.isfinite(max_qacc)
        assert max_qacc < 1e-3
    finally:
        backend.shutdown()


def test_oldroom_position_actuated_arm_twist_servo_is_stable() -> None:
    backend = MuJoCoBackend(str(OLDROOM_SCENE_PATH), headless=True)
    try:
        spec = backend.get_robot_spec()
        group_names = {jmg.name for jmg in spec.joint_model_groups}
        ee_names = {ee.name for jmg in spec.joint_model_groups for ee in jmg.end_effectors}
        for group_name, ee_name in (("left_arm", "left_ee"), ("right_arm", "right_ee")):
            command = core_pb2.ServoCommand(
                twist_cmd=core_pb2.TwistCommand(
                    twist=common_pb2.TwistStamped(),
                    target_ee=core_pb2.EESpec(name=ee_name, parent_jmg_name=group_name),
                )
            )
            command.twist_cmd.twist.twist.linear.z = 0.01

            states = backend.servo_control_stream(iter([command]))
            state = next(states)
            assert state.name
            assert group_name in group_names
            assert ee_name in ee_names
            time.sleep(0.15)

            command.twist_cmd.twist.twist.linear.z = 0.0
            next(backend.servo_control_stream(iter([command])))
            time.sleep(0.05)

            with backend._state_lock:
                arm_qacc = backend._data.qacc[12:28].copy()
            assert np.all(np.isfinite(arm_qacc))
            assert float(np.max(np.abs(arm_qacc))) < 10.0
    finally:
        backend.shutdown()


def test_bedroom_smart_car_wheels_start_stable() -> None:
    backend = MuJoCoBackend(str(BEDROOM_SCENE_PATH), headless=True)
    try:
        wheel_names = ("joint_fl", "joint_fr", "joint_bl", "joint_br")
        assert all(name not in backend._control_targets for name in wheel_names)

        time.sleep(0.6)
        with backend._state_lock:
            data = backend._data
            model = backend._model
            car_joint_id = model.joint("car_joint").id
            car_qpos_adr = int(model.jnt_qposadr[car_joint_id])
            wheel_dofs = [int(model.jnt_dofadr[model.joint(name).id]) for name in wheel_names]
            car_z = float(data.qpos[car_qpos_adr + 2])
            wheel_qvel = data.qvel[wheel_dofs].copy()
            max_qacc = float(np.max(np.abs(data.qacc)))

        assert car_z == pytest.approx(0.08, abs=5e-3)
        assert np.all(np.isfinite(wheel_qvel))
        assert float(np.max(np.abs(wheel_qvel))) < 0.05
        assert np.isfinite(max_qacc)
        assert max_qacc < 1e-2
    finally:
        backend.shutdown()


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
        abs(value) > 0.0 for value in (orientation.x, orientation.y, orientation.z, orientation.w)
    )
