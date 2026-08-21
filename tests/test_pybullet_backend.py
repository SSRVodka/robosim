"""Tests for the PyBullet backend."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pytest

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.robot_core_pb2 import ServoCommand
from control_stubs.robot_data_pb2 import RecordInfo, RecordOptions
from robosim.backends.pybullet.backend import PyBulletBackend
from robosim.core.impl.recorder_lerobot import LerobotDataRecorder

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"
SHARED_OPENUSD_CSD = FIXTURE_ROOT / "openusd" / "shared_tabletop" / "csd.usda"
SEMANTIC_OPENUSD_ROOT = FIXTURE_ROOT / "openusd" / "semantic"


def _csd_fixture(name: str) -> Path:
    return SEMANTIC_OPENUSD_ROOT / name.removesuffix(".json") / "csd.usda"


@pytest.fixture
def backend() -> Generator[PyBulletBackend, None, None]:
    instance = PyBulletBackend(headless=True)
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


def _image_array(image) -> np.ndarray:
    return np.frombuffer(image.data, dtype=np.uint8).reshape(
        int(image.height),
        int(image.width),
        3,
    )


def test_default_pybullet_backend_starts_and_renders(backend: PyBulletBackend) -> None:
    spec = backend.get_robot_spec()
    sensors = {entry.name: entry.type for entry in backend.list_sensors().entries}

    assert spec.robot_name == "panda"
    assert "panda_arm" in {group.name for group in spec.joint_model_groups}
    assert "joint_states" in sensors
    assert "world_camera" in sensors

    image = backend.get_sensors(["world_camera"]).images[0]

    assert image.width == 320
    assert image.height == 240
    assert float(_image_array(image).std()) > 1.0


def test_pybullet_backend_position_control_and_servo_stream(backend: PyBulletBackend) -> None:
    before = backend.get_robot_state()
    index = list(before.name).index("panda_joint1")
    target = float(before.position[index]) + 0.08

    backend.set_joint_target(
        ["panda_joint1"],
        [target],
        core_pb2.JointCommand.ControlMode.POSITION,
        "panda_arm",
    )

    assert _wait_for_condition(
        lambda: abs(float(backend.get_robot_state().position[index]) - target) < 0.03,
        timeout=2.0,
    )
    command_state = backend.get_joint_command_state()
    assert "panda_joint1" in command_state.name

    requests = iter(
        [
            ServoCommand(
                joint_cmd=core_pb2.JointCommand(
                    name=["panda_joint1"],
                    data=[target],
                    mode=core_pb2.JointCommand.ControlMode.POSITION,
                    group=core_pb2.JointModelGroupRequest(jmg_name="panda_arm"),
                )
            )
        ]
    )
    response = next(backend.servo_control_stream(requests))
    assert "panda_joint1" in response.name


def test_pybullet_backend_reports_end_effector_pose(backend: PyBulletBackend) -> None:
    state = backend.get_end_effector_state("panda_arm")

    assert state.pose_stamped.header.frame_id == "world"
    assert state.pose_stamped.pose.orientation.w != 0.0


def test_pybullet_backend_records_and_replays_lerobot_episode(
    tmp_path: Path,
    backend: PyBulletBackend,
) -> None:
    recorder = LerobotDataRecorder(tmp_path, backend)
    options = RecordOptions(
        repo_name="pybullet_demo",
        task_text="hold pose",
        fps=5,
        jmg_included=["panda_arm"],
        sensor_name_included=["world_camera"],
    )

    job = recorder.episode_start(options)
    time.sleep(0.25)
    end_status = recorder.episode_end()
    replay_status = recorder.episode_replay(RecordInfo(repo_name="pybullet_demo", episode_id=0))

    assert job.episode_id == 0
    assert end_status.code == 1
    assert replay_status.code == 1
