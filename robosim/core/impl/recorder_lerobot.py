from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

from control_stubs import common_pb2, sensing_pb2
from control_stubs.common_pb2 import JointState, Status
from control_stubs.robot_core_pb2 import EndEffectorState, RobotSpecification
from control_stubs.robot_data_pb2 import RecordJobInfo, RecordOptions
from control_stubs.sensing_pb2 import (
    CameraImage,
    ImuData,
    LidarScan,
    OdometryData,
    SensorData,
    SensorMetaList,
    WrenchData,
)
from robosim.core.backend import SimulatorBackend
from robosim.core.recorder import DataRecorder

DEFAULT_RECORD_FPS = 30
RGB_CHANNEL_NAMES = ["height", "width", "channels"]
XYZ_NAMES = ["x", "y", "z"]
QUAT_NAMES = ["x", "y", "z", "w"]


@dataclass(slots=True)
class CaptureSnapshot:
    robot_state: JointState
    joint_command_state: JointState
    end_effector_states: dict[str, EndEffectorState]
    sensor_data: SensorData


@dataclass(slots=True)
class CapturePlan:
    repo_name: str
    dataset_root: Path
    task_text: str
    fps: int
    joint_names: list[str]
    action_joint_names: list[str]
    end_effector_groups: list[str]
    sensor_names: list[str]
    features: dict[str, dict]
    episode_id: int


@dataclass(slots=True)
class RecordingSession:
    plan: CapturePlan
    dataset: LeRobotDataset
    stop_event: threading.Event
    thread: threading.Thread
    failure: Exception | None = None


class LerobotDataRecorder(DataRecorder):
    """Persist sampled backend state as a LeRobotDataset v3 dataset."""

    def __init__(self, repo_root: Path, backend: SimulatorBackend) -> None:
        self._repo_root = repo_root
        self._backend = backend
        self._datasets_root = repo_root / "data" / "lerobot"
        self._lock = threading.RLock()
        self._session: RecordingSession | None = None

    def record_episode_start(self, options: RecordOptions) -> RecordJobInfo:
        with self._lock:
            if self._session is not None:
                raise RuntimeError("recording is already in progress")

            plan, snapshot = self._build_plan(options)
            dataset = self._open_dataset(plan)
            first_frame = self._build_frame(plan, snapshot)
            dataset.add_frame(first_frame)

            stop_event = threading.Event()
            session = RecordingSession(
                plan=plan,
                dataset=dataset,
                stop_event=stop_event,
                thread=threading.Thread(
                    target=self._sampling_loop,
                    args=(plan, dataset, stop_event),
                    name=f"lerobot_record_{plan.repo_name}",
                    daemon=True,
                ),
            )
            self._session = session
            session.thread.start()

        return RecordJobInfo(
            status=Status(code=common_pb2.STATUS_SUCCESS, message="recording started"),
            episode_id=plan.episode_id,
        )

    def record_episode_end(self) -> Status:
        with self._lock:
            if self._session is None:
                raise RuntimeError("recording is not in progress")
            session = self._session

        session.stop_event.set()
        session.thread.join()

        try:
            if session.failure is not None:
                if session.dataset.has_pending_frames():
                    session.dataset.clear_episode_buffer()
                raise session.failure
            if session.dataset.has_pending_frames():
                session.dataset.save_episode()
            return Status(code=common_pb2.STATUS_SUCCESS, message="recording finished")
        finally:
            with contextlib.suppress(Exception):
                session.dataset.finalize()
            with self._lock:
                if self._session is session:
                    self._session = None

    def _build_plan(self, options: RecordOptions) -> tuple[CapturePlan, CaptureSnapshot]:
        repo_name = self._normalize_repo_name(options.repo_name)
        dataset_root = self._datasets_root / repo_name
        fps = int(options.fps) or DEFAULT_RECORD_FPS
        task_text = options.task_text or "unspecified"

        robot_state = self._backend.get_robot_state()
        robot_spec = self._backend.get_robot_spec()
        joint_names, end_effector_groups = self._select_joint_targets(
            options,
            robot_spec,
            robot_state,
        )
        sensor_names = self._select_sensor_names(options, self._backend.list_sensors())
        snapshot = self._capture_snapshot(robot_state, end_effector_groups, sensor_names)
        action_joint_names = self._select_action_joint_names(
            joint_names,
            snapshot.joint_command_state,
        )
        features = self._build_features(
            joint_names,
            action_joint_names,
            end_effector_groups,
            snapshot,
        )
        episode_id = self._next_episode_id(repo_name, dataset_root)
        return (
            CapturePlan(
                repo_name=repo_name,
                dataset_root=dataset_root,
                task_text=task_text,
                fps=fps,
                joint_names=joint_names,
                action_joint_names=action_joint_names,
                end_effector_groups=end_effector_groups,
                sensor_names=sensor_names,
                features=features,
                episode_id=episode_id,
            ),
            snapshot,
        )

    def _next_episode_id(self, repo_name: str, dataset_root: Path) -> int:
        if not dataset_root.exists():
            return 0
        dataset = LeRobotDataset(repo_id=repo_name, root=dataset_root)
        return dataset.meta.total_episodes

    def _open_dataset(self, plan: CapturePlan) -> LeRobotDataset:
        if not plan.dataset_root.exists():
            return LeRobotDataset.create(
                repo_id=plan.repo_name,
                root=plan.dataset_root,
                fps=plan.fps,
                robot_type=self._backend.robot_name,
                features=plan.features,
                use_videos=False,
            )

        dataset = LeRobotDataset.resume(repo_id=plan.repo_name, root=plan.dataset_root)
        if dataset.fps != plan.fps:
            raise ValueError(f"dataset fps mismatch: {dataset.fps} != {plan.fps}")
        if self._normalize_features(dataset.features) != self._normalize_features(
            {**plan.features, **DEFAULT_FEATURES}
        ):
            raise ValueError("existing dataset features do not match current record options")
        return dataset

    def _sampling_loop(
        self,
        plan: CapturePlan,
        dataset: LeRobotDataset,
        stop_event: threading.Event,
    ) -> None:
        deadline = time.monotonic() + (1.0 / plan.fps)
        while True:
            delay = max(0.0, deadline - time.monotonic())
            if stop_event.wait(delay):
                return
            try:
                snapshot = self._capture_snapshot(None, plan.end_effector_groups, plan.sensor_names)
                dataset.add_frame(self._build_frame(plan, snapshot))
            except Exception as exc:
                with self._lock:
                    if self._session is not None and self._session.dataset is dataset:
                        self._session.failure = exc
                stop_event.set()
                return
            deadline += 1.0 / plan.fps

    def _select_joint_targets(
        self,
        options: RecordOptions,
        robot_spec: RobotSpecification,
        robot_state: JointState,
    ) -> tuple[list[str], list[str]]:
        group_map = {group.name: group for group in robot_spec.joint_model_groups}
        if options.jmg_included or options.jmg_excluded:
            selected_groups = self._apply_filters(
                [group.name for group in robot_spec.joint_model_groups],
                list(options.jmg_included),
                list(options.jmg_excluded),
                "joint model group",
            )
            joint_names: list[str] = []
            seen: set[str] = set()
            for group_name in selected_groups:
                for joint_name in group_map[group_name].joint_names:
                    if joint_name not in seen:
                        seen.add(joint_name)
                        joint_names.append(joint_name)
            end_effector_groups = [
                group_name for group_name in selected_groups if group_map[group_name].end_effectors
            ]
            return joint_names, end_effector_groups

        end_effector_groups = [
            group.name for group in robot_spec.joint_model_groups if group.end_effectors
        ]
        return list(robot_state.name), end_effector_groups

    def _select_sensor_names(
        self,
        options: RecordOptions,
        sensor_meta_list: SensorMetaList,
    ) -> list[str]:
        non_joint_names = [
            entry.name for entry in sensor_meta_list.entries if entry.type != sensing_pb2.JOINT
        ]
        if not non_joint_names:
            return []
        return self._apply_filters(
            non_joint_names,
            list(options.sensor_name_included),
            list(options.sensor_name_excluded),
            "sensor",
        )

    def _apply_filters(
        self,
        available_names: list[str],
        included_names: list[str],
        excluded_names: list[str],
        label: str,
    ) -> list[str]:
        available_set = set(available_names)
        unknown_included = sorted(set(included_names) - available_set)
        if unknown_included:
            raise ValueError(f"unknown {label} names: {unknown_included}")
        unknown_excluded = sorted(set(excluded_names) - available_set)
        if unknown_excluded:
            raise ValueError(f"unknown {label} names: {unknown_excluded}")

        filtered_names = [name for name in available_names if name not in set(excluded_names)]
        if included_names:
            return [name for name in filtered_names if name in set(included_names)]
        return filtered_names

    def _capture_snapshot(
        self,
        robot_state: JointState | None,
        end_effector_groups: list[str],
        sensor_names: list[str],
    ) -> CaptureSnapshot:
        return CaptureSnapshot(
            robot_state=robot_state or self._backend.get_robot_state(),
            joint_command_state=self._backend.get_joint_command_state(),
            end_effector_states={
                group_name: self._backend.get_end_effector_state(group_name)
                for group_name in end_effector_groups
            },
            sensor_data=self._backend.get_sensors(sensor_names) if sensor_names else SensorData(),
        )

    def _select_action_joint_names(
        self,
        joint_names: list[str],
        joint_command_state: JointState,
    ) -> list[str]:
        command_names = set(joint_command_state.name)
        return [joint_name for joint_name in joint_names if joint_name in command_names]

    def _build_features(
        self,
        joint_names: list[str],
        action_joint_names: list[str],
        end_effector_groups: list[str],
        snapshot: CaptureSnapshot,
    ) -> dict[str, dict]:
        features: dict[str, dict] = {}

        if joint_names:
            features["observation.state"] = self._vector_feature(joint_names)
            features["observation.velocity"] = self._vector_feature(joint_names)
            features["observation.effort"] = self._vector_feature(joint_names)

        if action_joint_names:
            features["action.position"] = self._vector_feature(action_joint_names)
            features["action.velocity"] = self._vector_feature(action_joint_names)
            features["action.effort"] = self._vector_feature(action_joint_names)

        for group_name in end_effector_groups:
            self._require_end_effector(snapshot, group_name)
            features[
                f"observation.end_effectors.{group_name}.position"
            ] = self._named_vector_feature(XYZ_NAMES)
            features[
                f"observation.end_effectors.{group_name}.orientation"
            ] = self._named_vector_feature(QUAT_NAMES)

        for image_name, image in self._image_arrays(snapshot.sensor_data).items():
            features[f"observation.images.{image_name}"] = {
                "dtype": "image",
                "shape": image.shape,
                "names": RGB_CHANNEL_NAMES,
            }

        for imu_name in self._imu_map(snapshot.sensor_data):
            features[f"observation.imu.{imu_name}.orientation"] = self._named_vector_feature(
                QUAT_NAMES
            )
            features[
                f"observation.imu.{imu_name}.angular_velocity"
            ] = self._named_vector_feature(XYZ_NAMES)
            features[
                f"observation.imu.{imu_name}.linear_acceleration"
            ] = self._named_vector_feature(XYZ_NAMES)

        for lidar_name, lidar in self._lidar_map(snapshot.sensor_data).items():
            features[f"observation.lidar.{lidar_name}.ranges"] = self._vector_feature(
                [str(index) for index in range(len(lidar.ranges))]
            )
            if lidar.intensities:
                features[f"observation.lidar.{lidar_name}.intensities"] = self._vector_feature(
                    [str(index) for index in range(len(lidar.intensities))]
                )

        for odom_name in self._odometry_map(snapshot.sensor_data):
            features[f"observation.odometry.{odom_name}.position"] = self._named_vector_feature(
                XYZ_NAMES
            )
            features[
                f"observation.odometry.{odom_name}.orientation"
            ] = self._named_vector_feature(QUAT_NAMES)
            features[
                f"observation.odometry.{odom_name}.linear_velocity"
            ] = self._named_vector_feature(XYZ_NAMES)
            features[
                f"observation.odometry.{odom_name}.angular_velocity"
            ] = self._named_vector_feature(XYZ_NAMES)

        for force_name in self._wrench_map(list(snapshot.sensor_data.forces)):
            features[f"observation.force.{force_name}"] = self._named_vector_feature(XYZ_NAMES)
        for torque_name in self._wrench_map(list(snapshot.sensor_data.torques)):
            features[f"observation.torque.{torque_name}"] = self._named_vector_feature(XYZ_NAMES)

        return features

    def _build_frame(
        self,
        plan: CapturePlan,
        snapshot: CaptureSnapshot,
    ) -> dict[str, np.ndarray | str]:
        frame: dict[str, np.ndarray | str] = {"task": plan.task_text}

        if plan.joint_names:
            frame["observation.state"] = self._joint_vector(
                snapshot.robot_state, plan.joint_names, "position"
            )
            frame["observation.velocity"] = self._joint_vector(
                snapshot.robot_state, plan.joint_names, "velocity"
            )
            frame["observation.effort"] = self._joint_vector(
                snapshot.robot_state, plan.joint_names, "effort"
            )

        if plan.action_joint_names:
            frame["action.position"] = self._joint_vector(
                snapshot.joint_command_state, plan.action_joint_names, "position"
            )
            frame["action.velocity"] = self._joint_vector(
                snapshot.joint_command_state, plan.action_joint_names, "velocity"
            )
            frame["action.effort"] = self._joint_vector(
                snapshot.joint_command_state, plan.action_joint_names, "effort"
            )

        for group_name in plan.end_effector_groups:
            ee_state = self._require_end_effector(snapshot, group_name)
            pose = ee_state.pose_stamped.pose
            frame[f"observation.end_effectors.{group_name}.position"] = np.asarray(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float32,
            )
            frame[f"observation.end_effectors.{group_name}.orientation"] = np.asarray(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float32,
            )

        for image_name, image in self._image_arrays(snapshot.sensor_data).items():
            frame[f"observation.images.{image_name}"] = image

        for imu_name, imu in self._imu_map(snapshot.sensor_data).items():
            frame[f"observation.imu.{imu_name}.orientation"] = np.asarray(
                [
                    imu.orientation.x,
                    imu.orientation.y,
                    imu.orientation.z,
                    imu.orientation.w,
                ],
                dtype=np.float32,
            )
            frame[f"observation.imu.{imu_name}.angular_velocity"] = np.asarray(
                [
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            frame[f"observation.imu.{imu_name}.linear_acceleration"] = np.asarray(
                [
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ],
                dtype=np.float32,
            )

        for lidar_name, lidar in self._lidar_map(snapshot.sensor_data).items():
            frame[f"observation.lidar.{lidar_name}.ranges"] = np.asarray(
                lidar.ranges,
                dtype=np.float32,
            )
            if lidar.intensities:
                frame[f"observation.lidar.{lidar_name}.intensities"] = np.asarray(
                    lidar.intensities,
                    dtype=np.float32,
                )

        for odom_name, odom in self._odometry_map(snapshot.sensor_data).items():
            frame[f"observation.odometry.{odom_name}.position"] = np.asarray(
                [odom.pose.position.x, odom.pose.position.y, odom.pose.position.z],
                dtype=np.float32,
            )
            frame[f"observation.odometry.{odom_name}.orientation"] = np.asarray(
                [
                    odom.pose.orientation.x,
                    odom.pose.orientation.y,
                    odom.pose.orientation.z,
                    odom.pose.orientation.w,
                ],
                dtype=np.float32,
            )
            frame[f"observation.odometry.{odom_name}.linear_velocity"] = np.asarray(
                [odom.twist.linear.x, odom.twist.linear.y, odom.twist.linear.z],
                dtype=np.float32,
            )
            frame[f"observation.odometry.{odom_name}.angular_velocity"] = np.asarray(
                [odom.twist.angular.x, odom.twist.angular.y, odom.twist.angular.z],
                dtype=np.float32,
            )

        for force_name, force in self._wrench_map(list(snapshot.sensor_data.forces)).items():
            frame[f"observation.force.{force_name}"] = np.asarray(
                [force.vector.x, force.vector.y, force.vector.z],
                dtype=np.float32,
            )
        for torque_name, torque in self._wrench_map(list(snapshot.sensor_data.torques)).items():
            frame[f"observation.torque.{torque_name}"] = np.asarray(
                [torque.vector.x, torque.vector.y, torque.vector.z],
                dtype=np.float32,
            )

        return frame

    def _joint_vector(
        self,
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

    def _image_arrays(self, sensor_data: SensorData) -> dict[str, np.ndarray]:
        return {image.name: self._decode_image(image) for image in sensor_data.images}

    def _imu_map(self, sensor_data: SensorData) -> dict[str, ImuData]:
        return {imu.name: imu for imu in sensor_data.imus}

    def _lidar_map(self, sensor_data: SensorData) -> dict[str, LidarScan]:
        return {lidar.name: lidar for lidar in sensor_data.lidars}

    def _odometry_map(self, sensor_data: SensorData) -> dict[str, OdometryData]:
        return {odom.name: odom for odom in sensor_data.odometries}

    def _wrench_map(self, messages: list[WrenchData]) -> dict[str, WrenchData]:
        return {message.name: message for message in messages}

    def _require_end_effector(self, snapshot: CaptureSnapshot, group_name: str) -> EndEffectorState:
        ee_state = snapshot.end_effector_states.get(group_name)
        if ee_state is None:
            raise ValueError(f"missing end-effector state for group '{group_name}'")
        return ee_state

    def _decode_image(self, image: CameraImage) -> np.ndarray:
        channel_count = self._channel_count(image.encoding)
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
        return np.ascontiguousarray(array)

    def _channel_count(self, encoding: str) -> int:
        if encoding in {"rgb8", "bgr8"}:
            return 3
        if encoding in {"rgba8", "bgra8"}:
            return 4
        if encoding == "mono8":
            return 1
        raise ValueError(f"unsupported image encoding '{encoding}'")

    def _normalize_repo_name(self, repo_name: str) -> str:
        normalized = repo_name.strip()
        if not normalized:
            raise ValueError("repo_name must not be empty")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("repo_name must be a plain directory name")
        return normalized

    def _vector_feature(self, names: list[str]) -> dict[str, object]:
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
        }

    def _named_vector_feature(self, names: list[str]) -> dict[str, object]:
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
        }

    def _normalize_features(self, features: dict[str, dict]) -> dict[str, dict]:
        normalized: dict[str, dict] = {}
        for key, value in features.items():
            normalized_value = dict(value)
            if "shape" in normalized_value:
                normalized_value["shape"] = tuple(normalized_value["shape"])
            if "names" in normalized_value and normalized_value["names"] is not None:
                normalized_value["names"] = list(normalized_value["names"])
            normalized[key] = normalized_value
        return normalized
