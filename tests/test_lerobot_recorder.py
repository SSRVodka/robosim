from __future__ import annotations

import time
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from control_stubs.common_pb2 import Empty, JointState, Pose, PoseStamped, Quaternion
from control_stubs.mobility_ai_pb2 import NavGoal, TaskFeedback
from control_stubs.robot_core_pb2 import (
    EESpec,
    EndEffectorState,
    JointCommand,
    JointLimit,
    JointModelGroupSpec,
    RobotSpecification,
    ServoCommand,
)
from control_stubs.robot_data_pb2 import RecordInfo, RecordOptions
from control_stubs.sensing_pb2 import SensorData, SensorMetaList
from robosim.backends.mujoco.backend import MuJoCoBackend
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability
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


class ReplaySpyBackend(SimulatorBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[float], int, str | None]] = []

    @property
    def capabilities(self) -> Capability:
        return Capability.SERVO_CAPABLE

    @property
    def robot_name(self) -> str:
        return "spy_robot"

    @property
    def headless_mode(self) -> bool:
        return True

    def set_headless_mode(self, enabled: bool) -> None:
        del enabled

    def get_robot_state(self) -> JointState:
        return JointState(name=["joint_a", "joint_b"], position=[0.0, 0.0])

    def get_robot_spec(self) -> RobotSpecification:
        return RobotSpecification(
            robot_name=self.robot_name,
            joints=[
                JointLimit(name="joint_a", type="hinge", jmg_names=["arm", "all"]),
                JointLimit(name="joint_b", type="hinge", jmg_names=["arm", "all"]),
                JointLimit(name="joint_c", type="hinge", jmg_names=["all"]),
            ],
            joint_model_groups=[
                JointModelGroupSpec(
                    name="arm",
                    joint_names=["joint_a", "joint_b"],
                    end_effectors=[EESpec(name="tool", parent_jmg_name="arm", group_name="tool")],
                ),
                JointModelGroupSpec(
                    name="all",
                    joint_names=["joint_a", "joint_b", "joint_c"],
                    end_effectors=[EESpec(name="tool", parent_jmg_name="arm", group_name="tool")],
                ),
            ],
        )

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        self.calls.append((list(names), list(data), int(mode), group))

    def servo_control_stream(
        self,
        request_iterator: Iterator[ServoCommand],
    ) -> Iterator[JointState]:
        del request_iterator
        raise NotImplementedError

    def get_end_effector_state(self, group: str) -> EndEffectorState:
        del group
        return EndEffectorState(
            pose_stamped=PoseStamped(
                pose=Pose(orientation=Quaternion(w=1.0)),
            )
        )

    def get_joint_command_state(self) -> JointState:
        return JointState()

    def list_sensors(self) -> SensorMetaList:
        return SensorMetaList()

    def get_sensors(self, names: list[str]) -> SensorData:
        del names
        return SensorData()

    def stream_sensors(self, names: list[str]) -> Iterator[SensorData]:
        del names
        return iter(())

    def get_robot_pose_in_map(self) -> PoseStamped:
        return PoseStamped(pose=Pose(orientation=Quaternion(w=1.0)))

    def navigate_to(self, goal: NavGoal) -> Iterator[TaskFeedback]:
        del goal
        return iter(())

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params

    def emergency_stop(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class RecorderActionSpyBackend(SimulatorBackend):
    @property
    def capabilities(self) -> Capability:
        return Capability.SERVO_CAPABLE

    @property
    def robot_name(self) -> str:
        return "action_spy_robot"

    @property
    def headless_mode(self) -> bool:
        return True

    def set_headless_mode(self, enabled: bool) -> None:
        del enabled

    def get_robot_state(self) -> JointState:
        return JointState(
            name=["joint_a", "joint_b"],
            position=[1.0, 2.0],
            velocity=[0.0, 0.0],
            effort=[0.0, 0.0],
        )

    def get_robot_spec(self) -> RobotSpecification:
        return RobotSpecification(
            robot_name=self.robot_name,
            joints=[
                JointLimit(name="joint_a", type="hinge", jmg_names=["arm"]),
                JointLimit(name="joint_b", type="hinge", jmg_names=["arm"]),
            ],
            joint_model_groups=[
                JointModelGroupSpec(
                    name="arm",
                    joint_names=["joint_a", "joint_b"],
                    end_effectors=[EESpec(name="tool", parent_jmg_name="arm", group_name="tool")],
                ),
            ],
        )

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        del names, data, mode, group

    def servo_control_stream(
        self,
        request_iterator: Iterator[ServoCommand],
    ) -> Iterator[JointState]:
        del request_iterator
        raise NotImplementedError

    def get_end_effector_state(self, group: str) -> EndEffectorState:
        del group
        return EndEffectorState(
            pose_stamped=PoseStamped(
                pose=Pose(orientation=Quaternion(w=1.0)),
            )
        )

    def get_joint_command_state(self) -> JointState:
        return JointState(
            name=["joint_a", "joint_b"],
            position=[0.25, 0.75],
            velocity=[0.0, 0.0],
            effort=[0.0, 0.0],
        )

    def list_sensors(self) -> SensorMetaList:
        return SensorMetaList()

    def get_sensors(self, names: list[str]) -> SensorData:
        del names
        return SensorData()

    def stream_sensors(self, names: list[str]) -> Iterator[SensorData]:
        del names
        return iter(())

    def get_robot_pose_in_map(self) -> PoseStamped:
        return PoseStamped(pose=Pose(orientation=Quaternion(w=1.0)))

    def navigate_to(self, goal: NavGoal) -> Iterator[TaskFeedback]:
        del goal
        return iter(())

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params

    def emergency_stop(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


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

    first_job = recorder.episode_start(options)
    time.sleep(0.25)
    first_status = recorder.episode_end()

    second_job = recorder.episode_start(options)
    time.sleep(0.25)
    second_status = recorder.episode_end()

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
    assert tuple(sample["action"].shape) == (7,)
    assert tuple(sample["observation.end_effectors.panda_arm.position"].shape) == (3,)
    assert tuple(sample["observation.images.world_camera"].shape) == (3, 240, 320)


def test_robot_data_servicer_records_episode(tmp_path: Path, backend: MuJoCoBackend) -> None:
    servicer = RobotDataServicer(LerobotDataRecorder(tmp_path, backend))
    context = DummyContext()

    job = servicer.EpisodeStart(
        RecordOptions(
            repo_name="servicer_dataset",
            task_text="hold pose",
            fps=2,
            jmg_included=["panda_arm"],
            sensor_name_included=["world_camera"],
        ),
        context,
    )
    status = servicer.EpisodeEnd(Empty(), context)

    assert context.code is None
    assert job.status.code == 1
    assert job.episode_id == 0
    assert status.code == 1


def test_lerobot_recorder_records_joint_command_as_action(tmp_path: Path) -> None:
    recorder = LerobotDataRecorder(tmp_path, RecorderActionSpyBackend())
    options = RecordOptions(
        repo_name="command_dataset",
        task_text="hold pose",
        fps=5,
        jmg_included=["arm"],
    )

    recorder.episode_start(options)
    status = recorder.episode_end()

    assert status.code == 1

    dataset = LeRobotDataset(
        repo_id="command_dataset",
        root=tmp_path / "data" / "lerobot" / "command_dataset",
    )
    sample = dataset[0]

    assert sample["observation.state"].tolist() == pytest.approx([1.0, 2.0])
    assert sample["action"].tolist() == pytest.approx([0.25, 0.75])


def _create_replay_dataset(tmp_path: Path, repo_name: str) -> None:
    dataset_root = tmp_path / "data" / "lerobot" / repo_name
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=dataset_root,
        fps=20,
        robot_type="spy_robot",
        features={
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a", "joint_b"],
            },
        },
        use_videos=False,
    )
    dataset.add_frame(
        {
            "task": "replay task",
            "action": np.asarray([0.1, 0.2], dtype=np.float32),
        }
    )
    dataset.add_frame(
        {
            "task": "replay task",
            "action": np.asarray([0.3, 0.4], dtype=np.float32),
        }
    )
    dataset.save_episode()
    dataset.finalize()


def test_lerobot_recorder_replays_episode(tmp_path: Path) -> None:
    _create_replay_dataset(tmp_path, "replay_dataset")
    backend = ReplaySpyBackend()
    recorder = LerobotDataRecorder(tmp_path, backend)

    status = recorder.episode_replay(
        RecordInfo(repo_name="replay_dataset", episode_id=0)
    )

    assert status.code == 1
    assert len(backend.calls) == 2
    assert backend.calls[0][0] == ["joint_a", "joint_b"]
    assert backend.calls[0][1] == pytest.approx([0.1, 0.2])
    assert backend.calls[0][2] == int(JointCommand.ControlMode.POSITION)
    assert backend.calls[0][3] == "arm"
    assert backend.calls[1][0] == ["joint_a", "joint_b"]
    assert backend.calls[1][1] == pytest.approx([0.3, 0.4])
    assert backend.calls[1][2] == int(JointCommand.ControlMode.POSITION)
    assert backend.calls[1][3] == "arm"


def test_robot_data_servicer_replays_episode(tmp_path: Path) -> None:
    _create_replay_dataset(tmp_path, "servicer_replay_dataset")
    servicer = RobotDataServicer(LerobotDataRecorder(tmp_path, ReplaySpyBackend()))
    context = DummyContext()

    status = servicer.EpisodeReplay(
        RecordInfo(repo_name="servicer_replay_dataset", episode_id=0),
        context,
    )

    assert context.code is None
    assert status.code == 1
