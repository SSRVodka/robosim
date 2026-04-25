from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from control_stubs.common_pb2 import JointState
from control_stubs.robot_core_pb2 import JointModelGroupSpec, RobotSpecification
from control_stubs.sensing_pb2 import CameraImage
from robosim.core.backend import SimulatorBackend


def joint_state_vector(
    joint_state: JointState,
    joint_names: list[str],
    field_name: str,
) -> np.ndarray:
    values = getattr(joint_state, field_name)
    value_by_name = {
        name: float(values[index]) if index < len(values) else 0.0
        for index, name in enumerate(joint_state.name)
    }
    return np.asarray(
        [value_by_name.get(joint_name, 0.0) for joint_name in joint_names],
        dtype=np.float32,
    )


def resolve_joint_group(
    joint_names: list[str],
    robot_spec: RobotSpecification,
) -> str | None:
    exact_matches = [
        group
        for group in robot_spec.joint_model_groups
        if list(group.joint_names) == joint_names
    ]
    if len(exact_matches) == 1:
        return exact_matches[0].name
    if len(exact_matches) > 1:
        raise ValueError("joint group is ambiguous")

    containing_groups = [
        group
        for group in robot_spec.joint_model_groups
        if all(joint_name in group.joint_names for joint_name in joint_names)
    ]
    if not containing_groups:
        return None

    shortest_groups = shortest_groups_for_joints(containing_groups)
    if len(shortest_groups) == 1:
        return shortest_groups[0].name
    raise ValueError("joint group is ambiguous")


def shortest_groups_for_joints(
    groups: list[JointModelGroupSpec],
) -> list[JointModelGroupSpec]:
    min_width = min(len(group.joint_names) for group in groups)
    return [group for group in groups if len(group.joint_names) == min_width]


def decode_camera_image(image: CameraImage) -> np.ndarray:
    channel_count = channel_count_for_encoding(image.encoding)
    raw = np.frombuffer(image.data, dtype=np.uint8)
    row_width = int(image.step) if image.step else int(image.width) * channel_count
    array: np.ndarray = raw.reshape(int(image.height), row_width)
    array = array[:, : int(image.width) * channel_count]
    array = array.reshape(int(image.height), int(image.width), channel_count)

    if image.encoding == "bgr8":
        array = array[:, :, ::-1]
    elif image.encoding == "bgra8":
        array = array[:, :, [2, 1, 0, 3]]

    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.array(array, copy=True)


def channel_count_for_encoding(encoding: str) -> int:
    if encoding in {"rgb8", "bgr8"}:
        return 3
    if encoding in {"rgba8", "bgra8"}:
        return 4
    if encoding == "mono8":
        return 1
    raise ValueError(f"unsupported image encoding '{encoding}'")


@dataclass(slots=True)
class PolicyRuntimeSpec:
    action_joint_names: list[str]
    observation_joint_names: list[str]
    image_keys: list[str]
    group_name: str | None


class LerobotObservationAdapter:
    """Build the minimal LeRobot observation dict needed for policy inference."""

    def __init__(
        self,
        backend: SimulatorBackend,
        dataset_meta: LeRobotDatasetMetadata,
        policy_config: PreTrainedConfig,
        group_name: str | None,
    ) -> None:
        self._backend = backend
        self._dataset_meta = dataset_meta
        self._policy_config = policy_config
        self._runtime_spec = self._build_runtime_spec(group_name)

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return self._runtime_spec

    def capture_observation(self) -> dict[str, np.ndarray]:
        robot_state = self._backend.get_robot_state()
        observation: dict[str, np.ndarray] = {}

        if self._runtime_spec.observation_joint_names:
            observation["observation.state"] = joint_state_vector(
                robot_state,
                self._runtime_spec.observation_joint_names,
                "position",
            )

        if self._runtime_spec.image_keys:
            sensor_names = [
                key.removeprefix("observation.images.")
                for key in self._runtime_spec.image_keys
            ]
            images = self._backend.get_sensors(sensor_names).images
            image_map = {
                f"observation.images.{image.name}": decode_camera_image(image)
                for image in images
            }
            missing = sorted(set(self._runtime_spec.image_keys) - set(image_map))
            if missing:
                raise ValueError(f"missing required camera observations: {missing}")
            observation.update(image_map)

        return observation

    def _build_runtime_spec(self, group_name: str | None) -> PolicyRuntimeSpec:
        input_features = self._policy_config.input_features or {}
        state_keys = [
            key
            for key, feature in input_features.items()
            if feature.type is FeatureType.STATE
        ]
        unsupported_state_keys = sorted(
            key for key in state_keys if key != "observation.state"
        )
        if unsupported_state_keys:
            raise ValueError(
                f"unsupported policy state inputs: {unsupported_state_keys}"
            )

        env_keys = [
            key
            for key, feature in input_features.items()
            if feature.type is FeatureType.ENV
        ]
        if env_keys:
            raise ValueError(f"unsupported policy env inputs: {sorted(env_keys)}")

        image_keys = sorted(
            key
            for key, feature in input_features.items()
            if feature.type is FeatureType.VISUAL
        )

        if not image_keys and not state_keys:
            raise ValueError("policy must require at least one supported observation feature")

        action_feature = self._dataset_meta.features.get("action")
        if action_feature is None:
            raise ValueError("dataset is missing 'action' feature metadata")
        action_joint_names = list(action_feature.get("names") or [])
        if not action_joint_names:
            raise ValueError("dataset action feature is missing joint names")

        state_feature = self._dataset_meta.features.get("observation.state")
        observation_joint_names = list(state_feature.get("names") or []) if state_feature else []
        if state_keys and not observation_joint_names:
            raise ValueError("dataset is missing 'observation.state' joint names")

        robot_spec = self._backend.get_robot_spec()
        resolved_group = resolve_joint_group(action_joint_names, robot_spec)
        if group_name is not None:
            group = next(
                (entry for entry in robot_spec.joint_model_groups if entry.name == group_name),
                None,
            )
            if group is None:
                raise ValueError(f"unknown joint model group '{group_name}'")
            if list(group.joint_names) != action_joint_names:
                raise ValueError(
                    f"joint model group '{group_name}' does not match policy action joints"
                )
            resolved_group = group_name

        return PolicyRuntimeSpec(
            action_joint_names=action_joint_names,
            observation_joint_names=observation_joint_names,
            image_keys=image_keys,
            group_name=resolved_group,
        )
