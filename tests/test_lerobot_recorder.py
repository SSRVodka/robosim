from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path

import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from control_stubs.common_pb2 import Empty
from control_stubs.robot_data_pb2 import RecordOptions
from robosim.backends.mujoco.backend import MuJoCoBackend
from robosim.core.impl.recorder_lerobot import LerobotDataRecorder
from robosim.grpc_server.robot_data import RobotDataServicer

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)


class DummyContext:
    def __init__(self) -> None:
        self.code = None

    def set_code(self, code) -> None:
        self.code = code


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def test_lerobot_recorder_writes_and_resumes(tmp_path: Path, backend: MuJoCoBackend) -> None:
    recorder = LerobotDataRecorder(tmp_path, backend)
    options = RecordOptions(
        repo_name="demo_dataset",
        task_text="reach target",
        fps=5,
        jmg_included=["panda_arm"],
        sensor_name_included=["world_camera"],
    )

    first_job = recorder.record_episode_start(options)
    time.sleep(0.25)
    first_status = recorder.record_episode_end()

    second_job = recorder.record_episode_start(options)
    time.sleep(0.25)
    second_status = recorder.record_episode_end()

    assert first_job.episode_id == 0
    assert second_job.episode_id == 1
    assert first_status.code == 1
    assert second_status.code == 1

    dataset_root = tmp_path / "data" / "lerobot" / "demo_dataset"
    dataset = LeRobotDataset(repo_id="demo_dataset", root=dataset_root)
    sample = dataset[0]

    assert dataset.num_episodes == 2
    assert dataset.num_frames >= 2
    assert not (dataset_root / "images").exists()
    assert dataset.features["observation.state"]["names"] == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    assert tuple(sample["observation.state"].shape) == (7,)
    assert tuple(sample["action.position"].shape) == (7,)
    assert tuple(sample["observation.end_effectors.panda_arm.position"].shape) == (3,)
    assert tuple(sample["observation.images.world_camera"].shape) == (3, 240, 320)


def test_robot_data_servicer_records_episode(tmp_path: Path, backend: MuJoCoBackend) -> None:
    servicer = RobotDataServicer(LerobotDataRecorder(tmp_path, backend))
    context = DummyContext()

    job = servicer.RecordEpisodeStart(
        RecordOptions(
            repo_name="servicer_dataset",
            task_text="hold pose",
            fps=2,
            jmg_included=["panda_arm"],
            sensor_name_included=["world_camera"],
        ),
        context,
    )
    status = servicer.RecordEpisodeEnd(Empty(), context)

    assert context.code is None
    assert job.status.code == 1
    assert job.episode_id == 0
    assert status.code == 1
