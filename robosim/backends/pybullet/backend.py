"""PyBullet backend implementation."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pybullet as p
import pybullet_data

from control_stubs import common_pb2, mobility_ai_pb2, sensing_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.sensing_pb2 import SensorType
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability
from robosim.core.csd import CsdRealizationManifest

JOINT_SENSOR_NAME = "joint_states"
DEFAULT_TIMESTEP_SEC = 1.0 / 240.0
STREAM_INTERVAL_SEC = 0.05
DEFAULT_CAMERA_WIDTH = 320
DEFAULT_CAMERA_HEIGHT = 240
POSITION_FORCE = 240.0
VELOCITY_FORCE = 120.0
EE_DLS_DAMPING = 1e-4
EE_JOINT_VELOCITY_LIMIT = 0.5


@dataclass(slots=True)
class JointInfo:
    name: str
    index: int
    joint_type: int
    link_name: str
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float


@dataclass(slots=True)
class EndEffectorInfo:
    name: str
    parent_group: str
    group_name: str
    parent_link: str
    link_index: int


@dataclass(slots=True)
class JointModelGroup:
    name: str
    joint_names: list[str]
    named_states: dict[str, list[float]] = field(default_factory=dict)
    end_effectors: list[EndEffectorInfo] = field(default_factory=list)


@dataclass(slots=True)
class SensorInfo:
    name: str
    sensor_type: SensorType
    source: str
    camera: dict[str, object] | None = None


class PyBulletBackend(SimulatorBackend):
    """PyBullet backend for robot simulation and CSD realization packages."""

    def __init__(
        self,
        scene_path: str | None = None,
        scene_meta_path: str | None = None,
        headless: bool = True,
    ) -> None:
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._paused = False
        self._headless_mode = headless
        self._client_id = p.connect(p.DIRECT if headless else p.GUI)
        self._body_names: dict[str, int] = {}
        self._robot_body_id: int | None = None
        self._robot_name = "pybullet_scene"
        self._scene_path = Path(scene_path).resolve() if scene_path else None
        self._scene_meta_path = Path(scene_meta_path).resolve() if scene_meta_path else None
        self._scene_metadata: dict[str, object] = {}
        self._control_targets: dict[str, tuple[core_pb2.JointCommand.ControlMode, float]] = {}

        p.setTimeStep(DEFAULT_TIMESTEP_SEC, physicsClientId=self._client_id)
        p.setRealTimeSimulation(0, physicsClientId=self._client_id)
        if self._scene_path is None:
            self._load_default_scene()
        else:
            self._load_generated_scene(self._scene_path, self._scene_meta_path)

        self._joint_infos = self._build_joint_infos()
        self._joint_infos_by_name = {info.name: info for info in self._joint_infos}
        self._controllable_joint_names = [info.name for info in self._joint_infos]
        self._joint_groups = self._build_joint_groups()
        self._joint_to_groups = self._build_joint_to_groups()
        self._sensors = self._build_sensors()
        self._capabilities = self._detect_capabilities()
        self._set_default_configuration()
        self._set_idle_position_targets()

        self._step_thread = threading.Thread(
            target=self._simulation_loop,
            name="pybullet_backend_step",
            daemon=True,
        )
        self._step_thread.start()

    @property
    def body_names(self) -> dict[str, int]:
        return dict(self._body_names)

    @classmethod
    def from_csd_realization_manifest(
        cls,
        manifest: CsdRealizationManifest,
        *,
        headless: bool = True,
    ) -> "PyBulletBackend":
        if manifest.backend != "pybullet":
            raise ValueError(f"manifest backend must be pybullet, got {manifest.backend!r}")
        root = Path(manifest.root_path)
        scene_path = root / manifest.entry_file
        meta_path = root / "scene_meta.json"
        if not scene_path.is_file():
            raise FileNotFoundError(f"PyBullet realization entry file is missing: {scene_path}")
        return cls(str(scene_path), str(meta_path), headless=headless)

    @classmethod
    def from_csd_realization_manifest_file(
        cls,
        manifest_path: Path,
        *,
        headless: bool = True,
    ) -> "PyBulletBackend":
        manifest = CsdRealizationManifest.from_json_dict(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
        if manifest.backend != "pybullet":
            raise ValueError(f"manifest backend must be pybullet, got {manifest.backend!r}")
        root = Path(manifest_path).resolve().parent
        scene_path = root / manifest.entry_file
        meta_path = root / "scene_meta.json"
        if not scene_path.is_file():
            raise FileNotFoundError(f"PyBullet realization entry file is missing: {scene_path}")
        return cls(str(scene_path), str(meta_path), headless=headless)

    @property
    def capabilities(self) -> Capability:
        return self._capabilities

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def headless_mode(self) -> bool:
        return self._headless_mode

    def set_headless_mode(self, enabled: bool) -> None:
        if enabled != self._headless_mode:
            raise NotImplementedError("PyBullet headless mode cannot be changed after startup")

    def get_robot_state(self) -> common_pb2.JointState:
        if self._robot_body_id is None:
            return common_pb2.JointState(header=self._build_header("world"))
        with self._state_lock:
            names: list[str] = []
            positions: list[float] = []
            velocities: list[float] = []
            efforts: list[float] = []
            for info in self._joint_infos:
                state = p.getJointState(
                    self._robot_body_id,
                    info.index,
                    physicsClientId=self._client_id,
                )
                names.append(info.name)
                positions.append(float(state[0]))
                velocities.append(float(state[1]))
                efforts.append(float(state[3]))
            return common_pb2.JointState(
                header=self._build_header("world"),
                name=names,
                position=positions,
                velocity=velocities,
                effort=efforts,
            )

    def get_joint_command_state(self) -> common_pb2.JointState:
        state = self.get_robot_state()
        positions = list(state.position)
        velocities = [0.0] * len(state.name)
        efforts = [0.0] * len(state.name)
        for index, joint_name in enumerate(state.name):
            mode_target = self._control_targets.get(joint_name)
            if mode_target is None:
                continue
            mode, target = mode_target
            if mode == core_pb2.JointCommand.ControlMode.POSITION:
                positions[index] = target
            elif mode == core_pb2.JointCommand.ControlMode.VELOCITY:
                velocities[index] = target
            elif mode == core_pb2.JointCommand.ControlMode.TORQUE:
                efforts[index] = target
        return common_pb2.JointState(
            header=state.header,
            name=state.name,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )

    def get_robot_spec(self) -> core_pb2.RobotSpecification:
        joints = [
            core_pb2.JointLimit(
                name=info.name,
                type=self._joint_type_name(info.joint_type),
                jmg_names=self._joint_to_groups.get(info.name, []),
                lower_limit=info.lower_limit,
                upper_limit=info.upper_limit,
                velocity_limit=info.max_velocity,
                acceleration_limit=0.0,
                effort_limit=info.max_force,
            )
            for info in self._joint_infos
        ]
        groups = [
            core_pb2.JointModelGroupSpec(
                name=group.name,
                joint_names=group.joint_names,
                named_states=[
                    core_pb2.JointModelGroupNamedState(name=name, joint_values=values)
                    for name, values in sorted(group.named_states.items())
                ],
                end_effectors=[
                    core_pb2.EESpec(
                        name=ee.name,
                        parent_jmg_name=ee.parent_group,
                        group_name=ee.group_name,
                        parent_link=ee.parent_link,
                    )
                    for ee in group.end_effectors
                ],
            )
            for group in self._joint_groups.values()
        ]
        return core_pb2.RobotSpecification(
            robot_name=self._robot_name,
            joints=joints,
            joint_model_groups=groups,
        )

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: core_pb2.JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        if self._robot_body_id is None:
            raise NotImplementedError("Joint write requires a loaded robot")
        if len(names) != len(data):
            raise ValueError("Joint names and data length do not match")
        group_info = self._resolve_group(group, names)
        allowed = set(group_info.joint_names)
        indices: list[int] = []
        values: list[float] = []
        forces: list[float] = []
        with self._state_lock:
            for joint_name, target in zip(names, data, strict=True):
                if joint_name not in allowed:
                    raise ValueError(
                        f"Joint '{joint_name}' does not belong to group '{group_info.name}'"
                    )
                info = self._joint_infos_by_name[joint_name]
                indices.append(info.index)
                values.append(float(target))
                forces.append(info.max_force or POSITION_FORCE)
                self._control_targets[joint_name] = (mode, float(target))
            if mode == core_pb2.JointCommand.ControlMode.POSITION:
                p.setJointMotorControlArray(
                    self._robot_body_id,
                    indices,
                    p.POSITION_CONTROL,
                    targetPositions=values,
                    forces=forces,
                    physicsClientId=self._client_id,
                )
            elif mode == core_pb2.JointCommand.ControlMode.VELOCITY:
                p.setJointMotorControlArray(
                    self._robot_body_id,
                    indices,
                    p.VELOCITY_CONTROL,
                    targetVelocities=values,
                    forces=[max(force, VELOCITY_FORCE) for force in forces],
                    physicsClientId=self._client_id,
                )
            elif mode == core_pb2.JointCommand.ControlMode.TORQUE:
                p.setJointMotorControlArray(
                    self._robot_body_id,
                    indices,
                    p.TORQUE_CONTROL,
                    forces=values,
                    physicsClientId=self._client_id,
                )
            else:
                raise ValueError(f"Unsupported control mode: {mode}")

    def servo_control_stream(
        self,
        request_iterator: Iterator[core_pb2.ServoCommand],
    ) -> Iterator[common_pb2.JointState]:
        for request in request_iterator:
            if request.HasField("joint_cmd"):
                joint_cmd = request.joint_cmd
                self.set_joint_target(
                    list(joint_cmd.name),
                    list(joint_cmd.data),
                    joint_cmd.mode,
                    joint_cmd.group.jmg_name if joint_cmd.HasField("group") else None,
                )
            elif request.HasField("twist_cmd"):
                twist_cmd = request.twist_cmd
                parent_group = twist_cmd.target_ee.parent_jmg_name or twist_cmd.target_ee.group_name
                if not parent_group:
                    raise ValueError("Twist command must specify an end-effector parent group")
                joint_names, velocities = self._solve_end_effector_twist(
                    parent_group,
                    np.array(
                        [
                            twist_cmd.twist.twist.linear.x,
                            twist_cmd.twist.twist.linear.y,
                            twist_cmd.twist.twist.linear.z,
                        ],
                        dtype=np.float64,
                    ),
                    np.array(
                        [
                            twist_cmd.twist.twist.angular.x,
                            twist_cmd.twist.twist.angular.y,
                            twist_cmd.twist.twist.angular.z,
                        ],
                        dtype=np.float64,
                    ),
                )
                self.set_joint_target(
                    joint_names,
                    velocities,
                    core_pb2.JointCommand.ControlMode.VELOCITY,
                    parent_group,
                )
            yield self.get_robot_state()

    def get_end_effector_state(self, group: str) -> core_pb2.EndEffectorState:
        if self._robot_body_id is None:
            raise NotImplementedError("End-effector read requires a loaded robot")
        ee = self._get_end_effector(group)
        with self._state_lock:
            state = p.getLinkState(
                self._robot_body_id,
                ee.link_index,
                computeForwardKinematics=True,
                physicsClientId=self._client_id,
            )
        position = state[4]
        orientation = state[5]
        return core_pb2.EndEffectorState(
            pose_stamped=common_pb2.PoseStamped(
                header=self._build_header("world"),
                pose=common_pb2.Pose(
                    position=self._build_point(position),
                    orientation=self._build_quaternion_xyzw(orientation),
                ),
            )
        )

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        return sensing_pb2.SensorMetaList(
            entries=[
                sensing_pb2.SensorMetaList.SensorMeta(name=name, type=info.sensor_type)
                for name, info in sorted(self._sensors.items())
            ]
        )

    def get_sensors(self, names: list[str]) -> sensing_pb2.SensorData:
        requested = set(names) if names else set(self._sensors)
        images: list[sensing_pb2.CameraImage] = []
        joints: list[sensing_pb2.JointData] = []
        with self._state_lock:
            for name in sorted(requested):
                sensor = self._sensors.get(name)
                if sensor is None:
                    continue
                if sensor.source == "joint_state":
                    joints.append(
                        sensing_pb2.JointData(name=name, joint_states=self.get_robot_state())
                    )
                elif sensor.source == "camera" and sensor.camera is not None:
                    images.append(self._render_camera(sensor))
        return sensing_pb2.SensorData(images=images, joints=joints)

    def stream_sensors(self, names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        while not self._stop_event.is_set():
            yield self.get_sensors(names)
            time.sleep(STREAM_INTERVAL_SEC)

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        raise NotImplementedError("Navigation not supported for PyBullet")

    def navigate_to(
        self,
        goal: mobility_ai_pb2.NavGoal,
    ) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        del goal
        raise NotImplementedError("Navigation not supported for PyBullet")

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params
        with self._state_lock:
            if self._scene_path is None:
                p.resetSimulation(physicsClientId=self._client_id)
                self._body_names.clear()
                self._robot_body_id = None
                self._load_default_scene()
            else:
                p.resetSimulation(physicsClientId=self._client_id)
                self._body_names.clear()
                self._robot_body_id = None
                self._load_generated_scene(self._scene_path, self._scene_meta_path)
            self._joint_infos = self._build_joint_infos()
            self._joint_infos_by_name = {info.name: info for info in self._joint_infos}
            self._controllable_joint_names = [info.name for info in self._joint_infos]
            self._joint_groups = self._build_joint_groups()
            self._joint_to_groups = self._build_joint_to_groups()
            self._sensors = self._build_sensors()
            self._capabilities = self._detect_capabilities()
            self._set_default_configuration()
            self._set_idle_position_targets()
            self._paused = False

    def emergency_stop(self) -> None:
        if self._robot_body_id is None:
            return
        with self._state_lock:
            self._control_targets.clear()
            for info in self._joint_infos:
                p.setJointMotorControl2(
                    self._robot_body_id,
                    info.index,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=0.0,
                    physicsClientId=self._client_id,
                )

    def set_object_pose(self, object_name: str, pose: common_pb2.Pose) -> None:
        body_id = self._body_names.get(object_name)
        if body_id is None:
            raise ValueError(f"Unknown PyBullet body '{object_name}'")
        with self._state_lock:
            p.resetBasePositionAndOrientation(
                body_id,
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                physicsClientId=self._client_id,
            )

    def shutdown(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_step_thread") and self._step_thread.is_alive():
            self._step_thread.join(timeout=2.0)
        with self._state_lock:
            if p.isConnected(self._client_id):
                p.disconnect(self._client_id)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def step_physics(self) -> None:
        with self._state_lock:
            p.stepSimulation(physicsClientId=self._client_id)

    def _simulation_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            if not self._paused:
                with self._state_lock:
                    p.stepSimulation(physicsClientId=self._client_id)
            delay = DEFAULT_TIMESTEP_SEC - (time.time() - started)
            if delay > 0.0:
                time.sleep(delay)

    def _load_default_scene(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client_id)
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        self._robot_body_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        self._robot_name = "panda"
        self._body_names["panda"] = self._robot_body_id
        self._scene_metadata = {
            "cameras": [
                {
                    "name": "world_camera",
                    "position": [1.4, 0.0, 1.2],
                    "target": [0.0, 0.0, 0.45],
                    "up": [0.0, 0.0, 1.0],
                    "width": DEFAULT_CAMERA_WIDTH,
                    "height": DEFAULT_CAMERA_HEIGHT,
                }
            ]
        }

    def _load_generated_scene(self, scene_path: Path, scene_meta_path: Path | None) -> None:
        if scene_meta_path is None:
            scene_meta_path = scene_path.with_name("scene_meta.json")
        self._scene_metadata = json.loads(scene_meta_path.read_text(encoding="utf-8"))
        module_name = f"_robosim_pybullet_scene_{abs(hash(scene_path))}"
        spec = importlib.util.spec_from_file_location(module_name, scene_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import generated PyBullet scene: {scene_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        handles = module.load_scene(self._client_id)
        bodies = handles.get("bodies", {})
        if not isinstance(bodies, dict):
            raise RuntimeError("generated PyBullet scene did not return body mapping")
        self._body_names = {str(name): int(body_id) for name, body_id in bodies.items()}
        robot_name = str(self._scene_metadata.get("robot_name", ""))
        if robot_name and robot_name in self._body_names:
            self._robot_body_id = self._body_names[robot_name]
            self._robot_name = robot_name

    def _build_joint_infos(self) -> list[JointInfo]:
        if self._robot_body_id is None:
            return []
        joint_infos: list[JointInfo] = []
        joint_count = p.getNumJoints(
            self._robot_body_id,
            physicsClientId=self._client_id,
        )
        for joint_index in range(joint_count):
            info = p.getJointInfo(self._robot_body_id, joint_index, physicsClientId=self._client_id)
            joint_type = int(info[2])
            if joint_type not in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                continue
            joint_infos.append(
                JointInfo(
                    name=info[1].decode(),
                    index=joint_index,
                    joint_type=joint_type,
                    link_name=info[12].decode(),
                    lower_limit=float(info[8]),
                    upper_limit=float(info[9]),
                    max_force=float(info[10]),
                    max_velocity=float(info[11]),
                )
            )
        return joint_infos

    def _build_joint_groups(self) -> dict[str, JointModelGroup]:
        joint_names = [info.name for info in self._joint_infos]
        if self._robot_name == "panda" and {"panda_joint1", "panda_joint7"} <= set(joint_names):
            arm = [f"panda_joint{index}" for index in range(1, 8)]
            hand = ["panda_finger_joint1", "panda_finger_joint2"]
            ee = EndEffectorInfo(
                name="hand",
                parent_group="panda_arm",
                group_name="panda_hand",
                parent_link="panda_hand",
                link_index=self._link_index("panda_grasptarget"),
            )
            return {
                "panda_arm": JointModelGroup(
                    name="panda_arm",
                    joint_names=arm,
                    named_states={
                        "ready": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                    },
                    end_effectors=[ee],
                ),
                "panda_hand": JointModelGroup(name="panda_hand", joint_names=hand),
                "panda_arm_hand": JointModelGroup(name="panda_arm_hand", joint_names=arm + hand),
            }
        if joint_names:
            return {"all": JointModelGroup(name="all", joint_names=joint_names)}
        return {}

    def _build_joint_to_groups(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {name: [] for name in self._joint_infos_by_name}
        for group in self._joint_groups.values():
            for joint_name in group.joint_names:
                if joint_name in mapping:
                    mapping[joint_name].append(group.name)
        return mapping

    def _build_sensors(self) -> dict[str, SensorInfo]:
        sensors = {
            JOINT_SENSOR_NAME: SensorInfo(
                name=JOINT_SENSOR_NAME,
                sensor_type=SensorType.JOINT,
                source="joint_state",
            )
        }
        raw_cameras = self._scene_metadata.get("cameras", [])
        if not isinstance(raw_cameras, list):
            raw_cameras = []
        for raw_camera in raw_cameras:
            if isinstance(raw_camera, dict):
                name = str(raw_camera.get("name", "world_camera"))
                sensors[name] = SensorInfo(
                    name=name,
                    sensor_type=SensorType.CAMERA,
                    source="camera",
                    camera=raw_camera,
                )
        return sensors

    def _detect_capabilities(self) -> Capability:
        caps = Capability.SIMULATION_CONTROL | Capability.EMERGENCY_STOP
        if self._joint_infos:
            caps |= Capability.JOINT_READ | Capability.JOINT_WRITE | Capability.SENSOR_JOINT
        if any(group.end_effectors for group in self._joint_groups.values()):
            caps |= Capability.END_EFFECTOR_READ
        if any(sensor.sensor_type == SensorType.CAMERA for sensor in self._sensors.values()):
            caps |= Capability.SENSOR_CAMERA
        return caps

    def _set_default_configuration(self) -> None:
        if self._robot_body_id is None:
            return
        for group in self._joint_groups.values():
            ready = group.named_states.get("ready")
            if ready is None:
                continue
            for joint_name, value in zip(group.joint_names, ready, strict=True):
                info = self._joint_infos_by_name[joint_name]
                p.resetJointState(
                    self._robot_body_id,
                    info.index,
                    targetValue=float(value),
                    targetVelocity=0.0,
                    physicsClientId=self._client_id,
                )

    def _set_idle_position_targets(self) -> None:
        if self._robot_body_id is None:
            return
        state = self.get_robot_state()
        self._control_targets = {
            name: (core_pb2.JointCommand.ControlMode.POSITION, float(position))
            for name, position in zip(state.name, state.position, strict=True)
        }
        for name, position in zip(state.name, state.position, strict=True):
            info = self._joint_infos_by_name[name]
            p.setJointMotorControl2(
                self._robot_body_id,
                info.index,
                p.POSITION_CONTROL,
                targetPosition=float(position),
                force=info.max_force or POSITION_FORCE,
                physicsClientId=self._client_id,
            )

    def _render_camera(self, sensor: SensorInfo) -> sensing_pb2.CameraImage:
        assert sensor.camera is not None
        camera = sensor.camera
        width = int(cast(int | float | str, camera.get("width", DEFAULT_CAMERA_WIDTH)))
        height = int(cast(int | float | str, camera.get("height", DEFAULT_CAMERA_HEIGHT)))
        view = p.computeViewMatrix(
            cameraEyePosition=camera.get("position", [1.4, 0.0, 1.2]),
            cameraTargetPosition=camera.get("target", [0.0, 0.0, 0.4]),
            cameraUpVector=camera.get("up", [0.0, 0.0, 1.0]),
        )
        projection = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=10.0,
        )
        _w, _h, rgba, _depth, _seg = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client_id,
        )
        raw = bytes(rgba)
        rgb = bytes(channel for index, channel in enumerate(raw) if index % 4 != 3)
        return sensing_pb2.CameraImage(
            header=self._build_header(sensor.name),
            name=sensor.name,
            height=height,
            width=width,
            encoding="rgb8",
            is_bigendian=False,
            step=width * 3,
            data=rgb,
        )

    def _resolve_group(self, group: str | None, names: list[str]) -> JointModelGroup:
        if group is not None:
            group_info = self._joint_groups.get(group)
            if group_info is None:
                raise ValueError(f"Unknown joint model group '{group}'")
            return group_info
        candidates = [
            group_info
            for group_info in self._joint_groups.values()
            if all(name in group_info.joint_names for name in names)
        ]
        if len(candidates) != 1:
            raise ValueError("Joint group must be specified when the target names are ambiguous")
        return candidates[0]

    def _get_end_effector(self, group: str) -> EndEffectorInfo:
        group_info = self._joint_groups.get(group)
        if group_info is None or not group_info.end_effectors:
            raise NotImplementedError(f"Joint model group '{group}' has no end effector")
        return group_info.end_effectors[0]

    def _solve_end_effector_twist(
        self,
        parent_group: str,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> tuple[list[str], list[float]]:
        if self._robot_body_id is None:
            raise NotImplementedError("Twist servo requires a loaded robot")
        group = self._joint_groups[parent_group]
        ee = self._get_end_effector(parent_group)
        positions = [
            p.getJointState(self._robot_body_id, info.index, self._client_id)[0]
            for info in self._joint_infos
        ]
        velocities = [0.0] * len(positions)
        accelerations = [0.0] * len(positions)
        jac_t, jac_r = p.calculateJacobian(
            self._robot_body_id,
            ee.link_index,
            [0.0, 0.0, 0.0],
            positions,
            velocities,
            accelerations,
            physicsClientId=self._client_id,
        )
        joint_indices = [
            self._joint_infos.index(self._joint_infos_by_name[name])
            for name in group.joint_names
        ]
        jacobian = np.vstack((np.asarray(jac_t), np.asarray(jac_r)))[:, joint_indices]
        target = np.concatenate((linear, angular))
        lhs = jacobian.T @ jacobian + EE_DLS_DAMPING * np.eye(len(joint_indices))
        rhs = jacobian.T @ target
        qvel = np.linalg.solve(lhs, rhs)
        qvel = np.clip(qvel, -EE_JOINT_VELOCITY_LIMIT, EE_JOINT_VELOCITY_LIMIT)
        return group.joint_names, [float(value) for value in qvel]

    def _link_index(self, link_name: str) -> int:
        if self._robot_body_id is None:
            return -1
        joint_count = p.getNumJoints(
            self._robot_body_id,
            physicsClientId=self._client_id,
        )
        for joint_index in range(joint_count):
            info = p.getJointInfo(self._robot_body_id, joint_index, physicsClientId=self._client_id)
            if info[12].decode() == link_name:
                return joint_index
        return max(0, p.getNumJoints(self._robot_body_id, physicsClientId=self._client_id) - 1)

    def _joint_type_name(self, joint_type: int) -> str:
        if joint_type == p.JOINT_REVOLUTE:
            return "hinge"
        if joint_type == p.JOINT_PRISMATIC:
            return "slide"
        return "unknown"

    def _build_header(self, frame_id: str) -> common_pb2.Header:
        return common_pb2.Header(
            seq=0,
            timestamp=time.time(),
            frame_id=frame_id,
        )

    def _build_point(self, values: Any) -> common_pb2.Point:
        return common_pb2.Point(
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
        )

    def _build_quaternion_xyzw(self, values: Any) -> common_pb2.Quaternion:
        return common_pb2.Quaternion(
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
            w=float(values[3]),
        )
