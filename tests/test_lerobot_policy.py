from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from control_stubs.common_pb2 import JointState, Pose, PoseStamped, Quaternion
from control_stubs.mobility_ai_pb2 import NavGoal, TaskFeedback
from control_stubs.policy_pb2 import PolicyLoadRequest, PolicyStartRequest
from control_stubs.robot_core_pb2 import (
    EndEffectorState,
    JointCommand,
    JointLimit,
    JointModelGroupSpec,
    RobotSpecification,
    ServoCommand,
)
from control_stubs.sensing_pb2 import CameraImage, SensorData, SensorMetaList, SensorType
from robosim.backends.mujoco.backend import MuJoCoBackend
from robosim.backends.pybullet.backend import PyBulletBackend
from robosim.core.activity import ActivityCoordinator
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability
from robosim.core.impl.lerobot_io import LerobotObservationAdapter
from robosim.core.impl.policy_lerobot import LerobotPolicyRunner

PANDA_ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
PANDA_CAMERA_NAME = "world_camera"
MUJOCO_PANDA_SCENE = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim"
    / "mujoco"
    / "assets"
    / "robots"
    / "franka_panda"
    / "scene.xml"
)


def _create_dataset(
    repo_root: Path,
    repo_name: str,
    *,
    joint_names: list[str] | None = None,
    camera_name: str = "camera",
    robot_type: str = "test_robot",
    fps: int = 10,
) -> Path:
    joint_names = joint_names or ["joint_a", "joint_b"]
    joint_count = len(joint_names)
    image_key = f"observation.images.{camera_name}"
    dataset_root = repo_root / "data" / "lerobot" / repo_name
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=dataset_root,
        fps=fps,
        robot_type=robot_type,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (joint_count,),
                "names": joint_names,
            },
            image_key: {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": (joint_count,),
                "names": joint_names,
            },
        },
        use_videos=False,
    )
    dataset.add_frame(
        {
            "task": "test task",
            "observation.state": np.zeros(joint_count, dtype=np.float32),
            image_key: np.zeros((240, 320, 3), dtype=np.uint8),
            "action": np.zeros(joint_count, dtype=np.float32),
        }
    )
    dataset.save_episode()
    dataset.finalize()
    return dataset_root


def _create_policy_checkpoint(
    checkpoint_root: Path,
    *,
    joint_count: int = 2,
    camera_name: str = "camera",
) -> Path:
    image_key = f"observation.images.{camera_name}"
    config = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(joint_count,)),
            image_key: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 240, 320),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(joint_count,)),
        },
        n_action_steps=2,
        chunk_size=2,
        pretrained_backbone_weights=None,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        latent_dim=8,
        use_vae=False,
    )
    policy = ACTPolicy(config)
    stats = {
        "observation.state": {
            "mean": torch.zeros(joint_count),
            "std": torch.ones(joint_count),
        },
        image_key: {
            "mean": torch.zeros((3, 1, 1)),
            "std": torch.ones((3, 1, 1)),
        },
        "action": {
            "mean": torch.zeros(joint_count),
            "std": torch.ones(joint_count),
        },
    }
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(checkpoint_root)
    preprocessor.save_pretrained(checkpoint_root)
    postprocessor.save_pretrained(checkpoint_root)
    return checkpoint_root


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class PolicySpyBackend(SimulatorBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[float], int, str | None]] = []

    @property
    def capabilities(self) -> Capability:
        return Capability.SERVO_CAPABLE

    @property
    def robot_name(self) -> str:
        return "policy_spy_robot"

    @property
    def headless_mode(self) -> bool:
        return True

    def set_headless_mode(self, enabled: bool) -> None:
        del enabled

    def get_robot_state(self) -> JointState:
        return JointState(
            name=["joint_a", "joint_b"],
            position=[0.1, -0.2],
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
                JointModelGroupSpec(name="arm", joint_names=["joint_a", "joint_b"]),
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
            pose_stamped=PoseStamped(pose=Pose(orientation=Quaternion(w=1.0)))
        )

    def get_joint_command_state(self) -> JointState:
        return JointState(name=["joint_a", "joint_b"], position=[0.0, 0.0])

    def list_sensors(self) -> SensorMetaList:
        return SensorMetaList(
            entries=[SensorMetaList.SensorMeta(name="camera", type=SensorType.CAMERA)]
        )

    def get_sensors(self, names: list[str]) -> SensorData:
        assert names == ["camera"]
        image = np.full((240, 320, 3), 127, dtype=np.uint8)
        return SensorData(
            images=[
                CameraImage(
                    name="camera",
                    width=320,
                    height=240,
                    encoding="rgb8",
                    step=320 * 3,
                    data=image.tobytes(),
                )
            ]
        )

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


def test_lerobot_observation_adapter_builds_minimal_observation(tmp_path: Path) -> None:
    dataset_root = _create_dataset(tmp_path, "policy_dataset")
    checkpoint_root = _create_policy_checkpoint(tmp_path / "checkpoint")
    backend = PolicySpyBackend()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_root)
    dataset_meta = LeRobotDatasetMetadata("policy_dataset", root=dataset_root)

    adapter = LerobotObservationAdapter(backend, dataset_meta, policy_config, group_name=None)
    observation = adapter.capture_observation()

    assert sorted(observation) == ["observation.images.camera", "observation.state"]
    assert tuple(observation["observation.state"].shape) == (2,)
    assert tuple(observation["observation.images.camera"].shape) == (240, 320, 3)
    assert adapter.runtime_spec.action_joint_names == ["joint_a", "joint_b"]
    assert adapter.runtime_spec.group_name == "arm"


def test_lerobot_policy_runner_executes_control_loop(tmp_path: Path) -> None:
    _create_dataset(tmp_path, "policy_dataset")
    checkpoint_root = _create_policy_checkpoint(tmp_path / "checkpoint")
    backend = PolicySpyBackend()
    runner = LerobotPolicyRunner(tmp_path, backend, activity_coordinator=ActivityCoordinator())

    load_status = runner.load_policy(
        PolicyLoadRequest(
            policy_path=str(checkpoint_root),
            dataset_repo_name="policy_dataset",
            task_text="pick object",
            control_fps=20,
        )
    )
    assert load_status.code == 1

    start_status = runner.start_policy(PolicyStartRequest(control_fps=20))
    assert start_status.code == 1
    time.sleep(0.25)
    stop_status = runner.stop_policy()

    assert stop_status.code == 1
    assert backend.calls
    assert backend.calls[0][0] == ["joint_a", "joint_b"]
    assert backend.calls[0][2] == int(JointCommand.ControlMode.POSITION)
    assert backend.calls[0][3] == "arm"
    assert runner.get_status().running is False


def test_lerobot_policy_runner_defaults_control_fps_to_dataset_fps(
    tmp_path: Path,
) -> None:
    _create_dataset(tmp_path, "policy_dataset", fps=12)
    checkpoint_root = _create_policy_checkpoint(tmp_path / "checkpoint")
    runner = LerobotPolicyRunner(tmp_path, PolicySpyBackend())

    load_status = runner.load_policy(
        PolicyLoadRequest(
            policy_path=str(checkpoint_root),
            dataset_repo_name="policy_dataset",
        )
    )

    assert load_status.code == 1
    assert runner.get_status().control_fps == 12


@pytest.mark.parametrize(
    ("backend_name", "backend_factory"),
    [
        (
            "mujoco",
            lambda: MuJoCoBackend(str(MUJOCO_PANDA_SCENE), headless=True),
        ),
        ("pybullet", lambda: PyBulletBackend(headless=True)),
    ],
)
def test_lerobot_act_policy_runs_on_franka_backend(
    tmp_path: Path,
    backend_name: str,
    backend_factory: Callable[[], SimulatorBackend],
) -> None:
    repo_name = f"{backend_name}_panda_policy_dataset"
    _create_dataset(
        tmp_path,
        repo_name,
        joint_names=PANDA_ARM_JOINTS,
        camera_name=PANDA_CAMERA_NAME,
        robot_type="panda",
    )
    checkpoint_root = _create_policy_checkpoint(
        tmp_path / f"{backend_name}_checkpoint",
        joint_count=len(PANDA_ARM_JOINTS),
        camera_name=PANDA_CAMERA_NAME,
    )
    backend = backend_factory()
    calls: list[tuple[list[str], list[float], int, str | None]] = []
    original_set_joint_target = backend.set_joint_target

    def set_joint_target_spy(
        names: list[str],
        data: list[float],
        mode: JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        calls.append((list(names), list(data), int(mode), group))
        original_set_joint_target(names, data, mode, group)

    backend.set_joint_target = set_joint_target_spy  # type: ignore[method-assign]
    runner = LerobotPolicyRunner(
        tmp_path,
        backend,
        activity_coordinator=ActivityCoordinator(),
    )

    try:
        load_status = runner.load_policy(
            PolicyLoadRequest(
                policy_path=str(checkpoint_root),
                dataset_repo_name=repo_name,
                task_text="hold pose",
                jmg_name="panda_arm",
                control_fps=5,
            )
        )
        assert load_status.code == 1

        start_status = runner.start_policy(PolicyStartRequest(control_fps=5))
        assert start_status.code == 1
        assert _wait_until(
            lambda: bool(calls) or bool(runner.get_status().status.message),
            timeout=8.0,
        )
        stop_status = runner.stop_policy()

        assert stop_status.code == 1
        assert runner.get_status().status.message == ""
        assert calls
        names, data, mode, group = calls[0]
        assert names == PANDA_ARM_JOINTS
        assert len(data) == len(PANDA_ARM_JOINTS)
        assert all(np.isfinite(data))
        assert mode == int(JointCommand.ControlMode.POSITION)
        assert group == "panda_arm"
    finally:
        runner.shutdown()
        backend.shutdown()


@pytest.mark.parametrize(
    ("backend_name", "backend_factory"),
    [
        (
            "mujoco",
            lambda: MuJoCoBackend(str(MUJOCO_PANDA_SCENE), headless=True),
        ),
        ("pybullet", lambda: PyBulletBackend(headless=True)),
    ],
)
def test_lerobot_act_policy_runs_multistep_headless_on_franka_backend(
    tmp_path: Path,
    backend_name: str,
    backend_factory: Callable[[], SimulatorBackend],
) -> None:
    min_steps = 50
    repo_name = f"{backend_name}_panda_multistep_policy_dataset"
    _create_dataset(
        tmp_path,
        repo_name,
        joint_names=PANDA_ARM_JOINTS,
        camera_name=PANDA_CAMERA_NAME,
        robot_type="panda",
    )
    checkpoint_root = _create_policy_checkpoint(
        tmp_path / f"{backend_name}_multistep_checkpoint",
        joint_count=len(PANDA_ARM_JOINTS),
        camera_name=PANDA_CAMERA_NAME,
    )
    backend = backend_factory()
    observations: list[list[float]] = []
    commands: list[tuple[list[str], list[float], int, str | None]] = []
    original_get_robot_state = backend.get_robot_state
    original_set_joint_target = backend.set_joint_target

    def get_robot_state_spy() -> JointState:
        state = original_get_robot_state()
        observations.append(list(state.position))
        return state

    def set_joint_target_spy(
        names: list[str],
        data: list[float],
        mode: JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        commands.append((list(names), list(data), int(mode), group))
        original_set_joint_target(names, data, mode, group)

    backend.get_robot_state = get_robot_state_spy  # type: ignore[method-assign]
    backend.set_joint_target = set_joint_target_spy  # type: ignore[method-assign]
    runner = LerobotPolicyRunner(
        tmp_path,
        backend,
        activity_coordinator=ActivityCoordinator(),
    )

    try:
        load_status = runner.load_policy(
            PolicyLoadRequest(
                policy_path=str(checkpoint_root),
                dataset_repo_name=repo_name,
                task_text="hold pose",
                jmg_name="panda_arm",
            )
        )
        assert load_status.code == 1
        assert runner.get_status().control_fps == 10

        start_status = runner.start_policy(PolicyStartRequest())
        assert start_status.code == 1
        assert _wait_until(
            lambda: len(commands) >= min_steps
            or bool(runner.get_status().status.message),
            timeout=20.0,
        )
        stop_status = runner.stop_policy()

        assert stop_status.code == 1
        assert runner.get_status().status.message == ""
        assert len(observations) >= min_steps
        assert len(commands) >= min_steps
        for names, data, mode, group in commands[:min_steps]:
            assert names == PANDA_ARM_JOINTS
            assert len(data) == len(PANDA_ARM_JOINTS)
            assert all(np.isfinite(data))
            assert mode == int(JointCommand.ControlMode.POSITION)
            assert group == "panda_arm"
    finally:
        runner.shutdown()
        backend.shutdown()


def test_lerobot_policy_runner_resets_policy_on_world_reset(tmp_path: Path) -> None:
    _create_dataset(tmp_path, "policy_dataset")
    checkpoint_root = _create_policy_checkpoint(tmp_path / "checkpoint")
    runner = LerobotPolicyRunner(tmp_path, PolicySpyBackend())
    runner.load_policy(
        PolicyLoadRequest(
            policy_path=str(checkpoint_root),
            dataset_repo_name="policy_dataset",
        )
    )

    loaded = runner._loaded
    assert loaded is not None
    loaded.policy.reset = Mock()

    runner.notify_world_reset()

    loaded.policy.reset.assert_called_once()
