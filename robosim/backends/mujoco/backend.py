"""MuJoCo backend implementation."""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import mujoco
import mujoco.viewer
import numpy as np

from control_stubs import common_pb2 as common_pb2
from control_stubs import mobility_ai_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs import sensing_pb2 as sensing_pb2
from control_stubs.sensing_pb2 import SensorType
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability

JOINT_SENSOR_NAME = "joint_states"
DEFAULT_CAMERA_WIDTH = 320
DEFAULT_CAMERA_HEIGHT = 240
STREAM_INTERVAL_SEC = 0.05
POSITION_KP = 200.0
POSITION_KD = 30.0
VELOCITY_KP = 40.0
EE_DLS_DAMPING = 1e-4
ACTUATOR_MODE_TOL = 1e-9


@dataclass(slots=True)
class JointInfo:
    name: str
    joint_id: int
    joint_type: int
    body_id: int
    body_name: str
    qpos_adr: int
    qvel_adr: int
    qpos_dim: int
    qvel_dim: int
    lower_limit: float
    upper_limit: float
    controllable: bool
    actuator_id: int | None = None
    actuator_ctrlrange: tuple[float, float] | None = None
    joint_actuator_forcerange: tuple[float, float] | None = None


@dataclass(slots=True)
class EndEffectorInfo:
    name: str
    parent_group: str
    group_name: str
    parent_link: str
    body_name: str | None
    site_name: str | None


@dataclass(slots=True)
class JointModelGroup:
    name: str
    joint_names: list[str]
    named_states: dict[str, list[float]] = field(default_factory=dict)
    end_effectors: list[EndEffectorInfo] = field(default_factory=list)
    tip_body_name: str | None = None


@dataclass(slots=True)
class SensorInfo:
    name: str
    sensor_type: SensorType
    source: str
    source_id: int | None = None


class MuJoCoBackend(SimulatorBackend):
    """MuJoCo backend for robot simulation."""

    def __init__(self, scene_path: str, headless: bool = True) -> None:
        self._scene_path = Path(scene_path).resolve()
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._paused = False
        self._headless_mode = headless

        self._model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self._data = mujoco.MjData(self._model)
        self._gravity_data = mujoco.MjData(self._model)

        self._check_joint_types()
        self._srdf_path = self._find_srdf_path()
        self._robot_name = self._infer_robot_name()
        self._robot_root_body_id = self._find_robot_root_body()
        self._robot_body_ids = self._collect_robot_body_ids()
        self._joint_infos = self._build_joint_infos()
        self._joint_infos_by_name = {info.name: info for info in self._joint_infos}
        self._body_children = self._build_body_children()
        self._body_joint_name = self._build_body_joint_name_map()
        self._controllable_joint_names = [
            info.name for info in self._joint_infos if info.controllable
        ]
        self._control_targets: dict[str, tuple[core_pb2.JointCommand.ControlMode, float]] = {}
        self._position_velocity_targets: dict[str, float] = {}
        self._joint_groups = self._build_joint_groups()
        self._joint_to_groups = self._build_joint_to_groups()
        self._sensors = self._build_sensors()
        self._capabilities = self._detect_capabilities()

        self._viewer: mujoco.viewer.Handle | None = None
        self._renderers: dict[tuple[int, int, int], mujoco.Renderer] = {}
        if not self._headless_mode:
            self._viewer = self._launch_viewer()

        self._apply_default_configuration_locked()
        mujoco.mj_forward(self._model, self._data)
        self._set_idle_hold_targets_locked()
        self._step_thread = threading.Thread(
            target=self._simulation_loop,
            name="mujoco_backend_step",
            daemon=True,
        )
        self._step_thread.start()

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
        with self._state_lock:
            if self._headless_mode == enabled:
                return
            self._headless_mode = enabled
            if enabled:
                if self._viewer is not None:
                    self._viewer.close()
                    self._viewer = None
            else:
                self._viewer = self._launch_viewer()

    def _simulation_loop(self) -> None:
        while not self._stop_event.is_set():
            step_start = time.time()
            if not self._paused:
                with self._state_lock:
                    mujoco.mj_step1(self._model, self._data)
                    self._apply_controls_locked()
                    mujoco.mj_step2(self._model, self._data)
                    self._sync_viewer_locked()

            delay = self._model.opt.timestep - (time.time() - step_start)
            if delay > 0:
                time.sleep(delay)

    def _sync_viewer_locked(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            self._viewer = None
            return
        self._viewer.sync(state_only=True)

    def _check_joint_types(self) -> None:
        for joint_id in range(self._model.njnt):
            if int(self._model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_BALL):
                raise NotImplementedError("Ball joints are not supported yet")

    def _find_srdf_path(self) -> Path | None:
        candidates: list[Path] = []
        for xml_path in self._included_xml_paths(self._scene_path):
            candidate = xml_path.with_suffix(".srdf")
            if candidate.exists():
                candidates.append(candidate)
        direct_candidates = sorted(self._scene_path.parent.glob("*.srdf"))
        if direct_candidates:
            candidates.extend(direct_candidates)
        return candidates[0] if candidates else None

    def _included_xml_paths(self, xml_path: Path) -> list[Path]:
        xml_root = ET.parse(xml_path).getroot()
        includes = [xml_path]
        for include in xml_root.findall("include"):
            include_file = include.attrib.get("file")
            if include_file:
                nested_path = (xml_path.parent / include_file).resolve()
                includes.extend(self._included_xml_paths(nested_path))
        return includes

    def _infer_robot_name(self) -> str:
        if self._srdf_path is not None:
            srdf_root = ET.parse(self._srdf_path).getroot()
            srdf_name = srdf_root.attrib.get("name")
            if srdf_name:
                return srdf_name
        for xml_path in self._included_xml_paths(self._scene_path)[1:]:
            model_name = ET.parse(xml_path).getroot().attrib.get("model")
            if model_name:
                return model_name
        return self._model.body(self._find_robot_root_body()).name

    def _find_robot_root_body(self) -> int:
        best_body_id = -1
        best_joint_count = -1
        for body_id in range(1, self._model.nbody):
            if int(self._model.body_parentid[body_id]) != 0:
                continue
            body_ids = self._collect_subtree_body_ids(body_id)
            joint_count = sum(
                1
                for joint_id in range(self._model.njnt)
                if int(self._model.jnt_bodyid[joint_id]) in body_ids
                and self._is_robot_joint_type(int(self._model.jnt_type[joint_id]))
            )
            if joint_count > best_joint_count:
                best_body_id = body_id
                best_joint_count = joint_count
        if best_body_id < 0:
            raise RuntimeError("Unable to identify robot root body in MuJoCo scene")
        return best_body_id

    def _collect_robot_body_ids(self) -> set[int]:
        body_ids: set[int] = set()
        for body_id in range(1, self._model.nbody):
            if int(self._model.body_parentid[body_id]) != 0:
                continue
            subtree_ids = self._collect_subtree_body_ids(body_id)
            has_robot_joint = any(
                int(self._model.jnt_bodyid[joint_id]) in subtree_ids
                and self._is_robot_joint_type(int(self._model.jnt_type[joint_id]))
                for joint_id in range(self._model.njnt)
            )
            if has_robot_joint:
                body_ids.update(subtree_ids)
        if not body_ids:
            raise RuntimeError("Unable to identify robot bodies in MuJoCo scene")
        return body_ids

    def _collect_subtree_body_ids(self, body_id: int) -> set[int]:
        result = {body_id}
        for child_id in range(1, self._model.nbody):
            if int(self._model.body_parentid[child_id]) == body_id:
                result.update(self._collect_subtree_body_ids(child_id))
        return result

    def _build_body_children(self) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {body_id: [] for body_id in range(self._model.nbody)}
        for body_id in range(1, self._model.nbody):
            parent_id = int(self._model.body_parentid[body_id])
            children[parent_id].append(body_id)
        return children

    def _build_body_joint_name_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for info in self._joint_infos:
            mapping[info.body_name] = info.name
        return mapping

    def _build_joint_infos(self) -> list[JointInfo]:
        actuator_by_joint_id: dict[int, tuple[int, tuple[float, float], int | None]] = {}
        for actuator_id in range(self._model.nu):
            trn_type = int(self._model.actuator_trntype[actuator_id])
            if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            joint_id = int(self._model.actuator_trnid[actuator_id][0])
            ctrlrange = self._model.actuator_ctrlrange[actuator_id]
            actuator_by_joint_id[joint_id] = (
                actuator_id,
                (float(ctrlrange[0]), float(ctrlrange[1])),
                self._infer_actuator_control_mode(actuator_id),
            )

        joint_infos: list[JointInfo] = []
        for joint_id in range(self._model.njnt):
            body_id = int(self._model.jnt_bodyid[joint_id])
            if body_id not in self._robot_body_ids:
                continue
            joint_type = int(self._model.jnt_type[joint_id])
            qpos_dim, qvel_dim = self._joint_dims(joint_type)
            lower_limit = 0.0
            upper_limit = 0.0
            if joint_type in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                lower_limit = float(self._model.jnt_range[joint_id][0])
                upper_limit = float(self._model.jnt_range[joint_id][1])

            actuator = actuator_by_joint_id.get(joint_id)
            joint_actuator_forcerange = None
            if int(self._model.jnt_actfrclimited[joint_id]):
                force_range = self._model.jnt_actfrcrange[joint_id]
                joint_actuator_forcerange = (float(force_range[0]), float(force_range[1]))
            joint_infos.append(
                JointInfo(
                    name=self._model.joint(joint_id).name,
                    joint_id=joint_id,
                    joint_type=joint_type,
                    body_id=body_id,
                    body_name=self._model.body(body_id).name,
                    qpos_adr=int(self._model.jnt_qposadr[joint_id]),
                    qvel_adr=int(self._model.jnt_dofadr[joint_id]),
                    qpos_dim=qpos_dim,
                    qvel_dim=qvel_dim,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    controllable=self._is_robot_joint_type(joint_type),
                    actuator_id=actuator[0] if actuator else None,
                    actuator_ctrlrange=actuator[1] if actuator else None,
                    joint_actuator_forcerange=joint_actuator_forcerange,
                )
            )
        return joint_infos

    def _infer_actuator_control_mode(self, actuator_id: int) -> int | None:
        gain = self._model.actuator_gainprm[actuator_id]
        bias = self._model.actuator_biasprm[actuator_id]
        if abs(float(gain[0])) <= ACTUATOR_MODE_TOL:
            return None
        if (
            abs(float(bias[1]) + float(gain[0])) <= ACTUATOR_MODE_TOL
            and abs(float(bias[2])) <= ACTUATOR_MODE_TOL
        ):
            return int(core_pb2.JointCommand.ControlMode.POSITION)
        if (
            abs(float(bias[1])) <= ACTUATOR_MODE_TOL
            and abs(float(bias[2]) + float(gain[0])) <= ACTUATOR_MODE_TOL
        ):
            return int(core_pb2.JointCommand.ControlMode.VELOCITY)
        return None

    def _build_joint_groups(self) -> dict[str, JointModelGroup]:
        if self._srdf_path is None:
            return self._build_fallback_joint_groups()

        srdf_root = ET.parse(self._srdf_path).getroot()
        groups = {
            group_name: JointModelGroup(name=group_name, joint_names=joint_names)
            for group_name, joint_names in self._parse_srdf_groups(srdf_root).items()
        }
        for group_name, states in self._parse_srdf_group_states(srdf_root).items():
            group = groups.get(group_name)
            if group is None:
                continue
            for state_name, state_map in states.items():
                group.named_states[state_name] = [
                    float(state_map.get(joint_name, 0.0)) for joint_name in group.joint_names
                ]

        for group in groups.values():
            group.tip_body_name = self._infer_group_tip_body(group.joint_names)

        for ee in self._parse_srdf_end_effectors(srdf_root, groups):
            group = groups.get(ee.parent_group)
            if group is not None:
                group.end_effectors.append(ee)
        return groups

    def _build_fallback_joint_groups(self) -> dict[str, JointModelGroup]:
        tree_groups: dict[str, JointModelGroup] = {}
        for body_id in range(1, self._model.nbody):
            if int(self._model.body_parentid[body_id]) != 0:
                continue
            tree_body_ids = self._collect_subtree_body_ids(body_id)
            joint_names = [
                info.name
                for info in self._joint_infos
                if info.controllable and info.body_id in tree_body_ids
            ]
            if not joint_names:
                continue
            group_name = self._model.body(body_id).name
            tree_groups[group_name] = JointModelGroup(
                name=group_name,
                joint_names=joint_names,
                tip_body_name=self._infer_group_tip_body(joint_names),
            )
        return tree_groups

    def _build_joint_to_groups(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {name: [] for name in self._joint_infos_by_name}
        for group in self._joint_groups.values():
            for joint_name in group.joint_names:
                if joint_name in mapping:
                    mapping[joint_name].append(group.name)
        return mapping

    def _build_sensors(self) -> dict[str, SensorInfo]:
        sensors: dict[str, SensorInfo] = {
            JOINT_SENSOR_NAME: SensorInfo(
                name=JOINT_SENSOR_NAME,
                sensor_type=SensorType.JOINT,
                source="joint_state",
            )
        }
        for sensor_id in range(self._model.nsensor):
            sensor = self._model.sensor(sensor_id)
            sensor_type = self._map_sensor_type(int(self._model.sensor_type[sensor_id]))
            if sensor_type is None:
                continue
            sensors[sensor.name] = SensorInfo(
                name=sensor.name,
                sensor_type=sensor_type,
                source="sensor",
                source_id=sensor_id,
            )
        for camera_id in range(self._model.ncam):
            camera = self._model.camera(camera_id)
            sensors[camera.name] = SensorInfo(
                name=camera.name,
                sensor_type=SensorType.CAMERA,
                source="camera",
                source_id=camera_id,
            )
        return sensors

    def _detect_capabilities(self) -> Capability:
        caps = (
            Capability.JOINT_READ
            | Capability.JOINT_WRITE
            | Capability.SIMULATION_CONTROL
            | Capability.EMERGENCY_STOP
        )
        if any(group.end_effectors for group in self._joint_groups.values()):
            caps |= Capability.END_EFFECTOR_READ
        for sensor in self._sensors.values():
            if sensor.sensor_type == SensorType.CAMERA:
                caps |= Capability.SENSOR_CAMERA
            elif sensor.sensor_type == SensorType.JOINT:
                caps |= Capability.SENSOR_JOINT
            elif sensor.sensor_type == SensorType.IMU:
                caps |= Capability.SENSOR_IMU
        return caps

    def get_robot_state(self) -> common_pb2.JointState:
        with self._state_lock:
            names: list[str] = []
            positions: list[float] = []
            velocities: list[float] = []
            efforts: list[float] = []
            for info in self._joint_infos:
                if not info.controllable:
                    continue
                names.append(info.name)
                positions.append(float(self._data.qpos[info.qpos_adr]))
                velocities.append(float(self._data.qvel[info.qvel_adr]))
                efforts.append(float(self._data.qfrc_applied[info.qvel_adr]))
            return common_pb2.JointState(
                header=self._build_header(frame_id=self._model.body(self._robot_root_body_id).name),
                name=names,
                position=positions,
                velocity=velocities,
                effort=efforts,
            )

    def get_joint_command_state(self) -> common_pb2.JointState:
        with self._state_lock:
            names = list(self._controllable_joint_names)
            # NOTE: DataRecorder will always use position.
            # So if the current control mode is not POSITION,
            # we need to convert the target to position.
            position_by_name = {
                info.name: float(self._data.qpos[info.qpos_adr])
                for info in self._joint_infos
                if info.controllable
            }
            positions = [position_by_name[name] for name in names]
            velocities = [0.0] * len(names)
            efforts = [0.0] * len(names)
            for index, joint_name in enumerate(names):
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
                header=self._build_header(frame_id=self._model.body(self._robot_root_body_id).name),
                name=names,
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
                velocity_limit=0.0,
                acceleration_limit=0.0,
                effort_limit=0.0,
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
        if len(names) != len(data):
            raise ValueError("Joint names and data length do not match")
        group_info = self._resolve_group(group, names)
        allowed = set(group_info.joint_names)
        with self._state_lock:
            for joint_name, target in zip(names, data, strict=True):
                if joint_name not in allowed:
                    raise ValueError(
                        f"Joint '{joint_name}' does not belong to group '{group_info.name}'"
                    )
                info = self._joint_infos_by_name.get(joint_name)
                if info is None or not info.controllable:
                    raise ValueError(f"Joint '{joint_name}' is not controllable")
                previous = self._control_targets.get(joint_name)
                if (
                    mode == core_pb2.JointCommand.ControlMode.VELOCITY
                    and info.actuator_control_mode
                    == int(core_pb2.JointCommand.ControlMode.POSITION)
                ):
                    if previous is None or previous[0] != mode:
                        self._position_velocity_targets[joint_name] = (
                            self._clamp_actuator_target(
                                info,
                                float(self._data.qpos[info.qpos_adr]),
                            )
                        )
                else:
                    self._position_velocity_targets.pop(joint_name, None)
                self._control_targets[joint_name] = (mode, float(target))

    def servo_control_stream(
        self, request_iterator: Iterator[core_pb2.ServoCommand]
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
                # Convert Cartesian world-frame velocities into joint-space velocities.
                joint_names, joint_velocities = self._solve_end_effector_twist(
                    parent_group=parent_group,
                    linear=np.array(
                        [
                            twist_cmd.twist.twist.linear.x,
                            twist_cmd.twist.twist.linear.y,
                            twist_cmd.twist.twist.linear.z,
                        ],
                        dtype=np.float64,
                    ),
                    angular=np.array(
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
                    joint_velocities,
                    core_pb2.JointCommand.ControlMode.VELOCITY,
                    parent_group,
                )
            yield self.get_robot_state()

    def get_end_effector_state(self, parent_group: str) -> core_pb2.EndEffectorState:
        ee = self._get_end_effector(parent_group)
        with self._state_lock:
            pose = self._get_end_effector_pose_locked(ee)
        return core_pb2.EndEffectorState(
            pose_stamped=common_pb2.PoseStamped(
                header=self._build_header(frame_id="world"),
                pose=pose,
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
        forces: list[sensing_pb2.WrenchData] = []
        torques: list[sensing_pb2.WrenchData] = []

        with self._state_lock:
            for name in sorted(requested):
                sensor = self._sensors.get(name)
                if sensor is None:
                    continue
                if sensor.source == "joint_state":
                    joints.append(
                        sensing_pb2.JointData(name=name, joint_states=self.get_robot_state())
                    )
                    continue
                if sensor.source == "camera":
                    images.append(self._render_camera_locked(sensor))
                    continue
                if sensor.source == "sensor" and sensor.source_id is not None:
                    sensor_slice = self._sensor_values(sensor.source_id)
                    header = self._build_header(frame_id="world")
                    if sensor.sensor_type == SensorType.LIDAR:
                        lidars.append(self._build_rangefinder_lidar_locked(sensor, sensor_slice))
                    elif sensor.sensor_type == SensorType.FORCE:
                        forces.append(
                            sensing_pb2.WrenchData(
                                header=header,
                                name=name,
                                vector=self._build_point(sensor_slice),
                            )
                        )
                    elif sensor.sensor_type == SensorType.TORQUE:
                        torques.append(
                            sensing_pb2.WrenchData(
                                header=header,
                                name=name,
                                vector=self._build_point(sensor_slice),
                            )
                        )

        return sensing_pb2.SensorData(images=images, joints=joints, forces=forces, torques=torques)

    def stream_sensors(self, names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        while not self._stop_event.is_set():
            yield self.get_sensors(names)
            time.sleep(STREAM_INTERVAL_SEC)

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params
        with self._state_lock:
            mujoco.mj_resetData(self._model, self._data)
            self._paused = False
            self._apply_default_configuration_locked()
            mujoco.mj_forward(self._model, self._data)
            self._set_idle_hold_targets_locked()
            self._sync_viewer_locked()

    def emergency_stop(self) -> None:
        with self._state_lock:
            self._control_targets.clear()
            self._position_velocity_targets.clear()
            self._data.ctrl[:] = 0
            self._data.qfrc_applied[:] = 0
            self._data.xfrc_applied[:] = 0

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._step_thread.is_alive():
            self._step_thread.join(timeout=2.0)
        with self._state_lock:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            for renderer in self._renderers.values():
                renderer.close()
            self._renderers.clear()

    # def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
    #     raise NotImplementedError("Navigation not supported for MuJoCo")
    # Eddy:Extension of Navigation which is based on two_bedroom_apartment
    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        with self._state_lock:
            body_id = self._robot_root_body_id
            position = self._data.xpos[body_id].copy()
            orientation = self._data.xquat[body_id].copy()

        return common_pb2.PoseStamped(
            header=self._build_header(frame_id="world"),
            pose=common_pb2.Pose(
                position=self._build_point(position),
                orientation=self._build_quaternion(orientation),
            ),
        )

    def navigate_to(
        self,
        goal: mobility_ai_pb2.NavGoal,
    ) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        if not self._supports_robot_vacuum_navigation():
            raise NotImplementedError("Navigation requires robot vacuum lidar and wheel joints")

        target_frame = goal.target_frame or "world"
        if target_frame not in ("world", "map"):
            raise ValueError(f"Unsupported navigation target_frame: {target_frame}")

        from robosim.navigation.backend_client import BackendRobosimClient
        from robosim.navigation.geometry import Point2D
        from robosim.navigation.mujoco_map import build_occupancy_grid
        from robosim.navigation.path_follower import (
            NavigationOutcome,
            navigate_with_lidar_safety,
        )

        task_id = f"mujoco-nav-{int(time.time() * 1000)}"
        target = Point2D(
            float(goal.target_pose.position.x),
            float(goal.target_pose.position.y),
        )
        max_velocity = float(goal.max_velocity)
        params = self._build_navigation_params(max_velocity)
        grid_options = self._build_navigation_grid_options()

        yield self._make_feedback(
            task_id=task_id,
            status_code=common_pb2.STATUS_RUNNING,
            message="Building navigation grid",
            feedback_text=(f"target=({target.x:.3f}, {target.y:.3f}), frame={target_frame}"),
        )

        built_grid = build_occupancy_grid(self._scene_path, grid_options)
        yield self._make_feedback(
            task_id=task_id,
            status_code=common_pb2.STATUS_RUNNING,
            message="Navigation started",
            feedback_text=(
                f"grid={built_grid.grid.width}x{built_grid.grid.height}, "
                f"ignored={built_grid.stats.skipped_ignored_geoms}"
            ),
        )

        feedback_queue: queue.Queue[
            mobility_ai_pb2.TaskFeedback | NavigationOutcome | BaseException
        ] = queue.Queue()
        cancel_event = threading.Event()
        nav_client = BackendRobosimClient(self)
        last_feedback_at = 0.0
        last_state = ""
        last_replans = -1

        def on_step(step: NavigationStep) -> None:
            nonlocal last_feedback_at, last_state, last_replans
            now = time.monotonic()
            if (
                now - last_feedback_at < NAV_FEEDBACK_INTERVAL_SEC
                and step.state == last_state
                and step.replans == last_replans
            ):
                return
            last_feedback_at = now
            last_state = step.state
            last_replans = step.replans
            feedback_queue.put(self._make_navigation_step_feedback(task_id, step))

        def worker() -> None:
            try:
                outcome = navigate_with_lidar_safety(
                    client=nav_client,
                    grid=built_grid.grid,
                    target=target,
                    params=params,
                    feedback=on_step,
                    should_cancel=cancel_event.is_set,
                )
            except BaseException as exc:  # noqa: BLE001
                feedback_queue.put(exc)
            else:
                feedback_queue.put(outcome)

        nav_thread = threading.Thread(
            target=worker,
            name="mujoco_navigation_worker",
            daemon=True,
        )
        nav_thread.start()

        try:
            while True:
                try:
                    item = feedback_queue.get(timeout=0.1)
                except queue.Empty:
                    if not nav_thread.is_alive():
                        yield self._make_feedback(
                            task_id=task_id,
                            status_code=common_pb2.STATUS_FAILURE,
                            message="Navigation worker exited without outcome",
                        )
                        break
                    continue

                if isinstance(item, mobility_ai_pb2.TaskFeedback):
                    yield item
                    continue
                if isinstance(item, NavigationOutcome):
                    yield self._make_navigation_outcome_feedback(task_id, item)
                    break
                raise item
        finally:
            if nav_thread.is_alive():
                cancel_event.set()
                nav_thread.join(timeout=2.0)
                self.emergency_stop()

    def _build_navigation_params(self, max_velocity: float) -> "NavigationParams":
        from robosim.navigation.path_follower import NavigationParams

        normal_linear = _env_float("ROBOSIM_NAV_NORMAL_LINEAR", 0.14)
        if max_velocity > 0.0:
            normal_linear = min(max_velocity, _env_float("ROBOSIM_NAV_MAX_LINEAR", 0.30))
        slow_linear = min(
            _env_float("ROBOSIM_NAV_SLOW_LINEAR", 0.04),
            max(0.02, normal_linear * 0.5),
        )
        return NavigationParams(
            timeout=_env_float("ROBOSIM_NAV_TIMEOUT", 60.0),
            arrive_tolerance=_env_float("ROBOSIM_NAV_ARRIVE_TOLERANCE", 0.12),
            waypoint_tolerance=_env_float("ROBOSIM_NAV_WAYPOINT_TOLERANCE", 0.18),
            lookahead_distance=_env_float("ROBOSIM_NAV_LOOKAHEAD_DISTANCE", 0.35),
            direct_goal_distance=_env_float("ROBOSIM_NAV_DIRECT_GOAL_DISTANCE", 0.55),
            stop_distance=_env_float("ROBOSIM_NAV_STOP_DISTANCE", 0.30),
            slow_distance=_env_float("ROBOSIM_NAV_SLOW_DISTANCE", 0.75),
            side_distance=_env_float("ROBOSIM_NAV_SIDE_DISTANCE", 0.40),
            normal_linear=normal_linear,
            slow_linear=slow_linear,
            max_angular=_env_float("ROBOSIM_NAV_MAX_ANGULAR", 1.2),
            goal_gain=_env_float("ROBOSIM_NAV_GOAL_GAIN", 1.6),
            avoid_gain=_env_float("ROBOSIM_NAV_AVOID_GAIN", 1.0),
            turn_speed=_env_float("ROBOSIM_NAV_TURN_SPEED", 0.8),
            replan_after=_env_float("ROBOSIM_NAV_REPLAN_AFTER", 0.8),
            replan_cooldown=_env_float("ROBOSIM_NAV_REPLAN_COOLDOWN", 4.0),
            progress_timeout=_env_float("ROBOSIM_NAV_PROGRESS_TIMEOUT", 3.0),
            progress_epsilon=_env_float("ROBOSIM_NAV_PROGRESS_EPSILON", 0.03),
            dynamic_mark_distance=_env_float("ROBOSIM_NAV_DYNAMIC_MARK_DISTANCE", 1.2),
            dynamic_min_distance=_env_float("ROBOSIM_NAV_DYNAMIC_MIN_DISTANCE", 0.22),
            dynamic_mark_half_angle_deg=_env_float(
                "ROBOSIM_NAV_DYNAMIC_MARK_HALF_ANGLE_DEG",
                120.0,
            ),
            dynamic_obstacle_radius=_env_float(
                "ROBOSIM_NAV_DYNAMIC_OBSTACLE_RADIUS",
                0.45,
            ),
        )

    def _build_navigation_grid_options(self) -> "GridBuildOptions":
        from robosim.navigation.mujoco_map import GridBuildOptions

        return GridBuildOptions(
            resolution=_env_float("ROBOSIM_NAV_GRID_RESOLUTION", 0.10),
            robot_radius=_env_float("ROBOSIM_NAV_ROBOT_RADIUS", 0.18),
            safety_margin=_env_float("ROBOSIM_NAV_SAFETY_MARGIN", 0.08),
            bounds_padding=_env_float("ROBOSIM_NAV_BOUNDS_PADDING", 0.20),
            bounds=_env_bounds("ROBOSIM_NAV_BOUNDS"),
            ignored_geom_names=_env_csv("ROBOSIM_NAV_IGNORE_GEOM_NAMES"),
            ignored_geom_prefixes=_env_csv("ROBOSIM_NAV_IGNORE_GEOM_PREFIXES"),
        )

    @staticmethod
    def _make_feedback(
        task_id: str,
        status_code: common_pb2.StatusCode,
        message: str,
        eta: int = 0,
        feedback_text: str = "",
    ) -> mobility_ai_pb2.TaskFeedback:
        feedback = mobility_ai_pb2.TaskFeedback()
        feedback.task_id = task_id
        feedback.status.code = status_code
        feedback.status.message = message
        feedback.eta = eta
        feedback.feedback_text = feedback_text
        return feedback

    def _make_navigation_step_feedback(
        self,
        task_id: str,
        step: "NavigationStep",
    ) -> mobility_ai_pb2.TaskFeedback:
        speed_for_eta = max(abs(step.linear), 0.05)
        eta = int(max(0.0, step.goal_distance / speed_for_eta))
        return self._make_feedback(
            task_id=task_id,
            status_code=common_pb2.STATUS_RUNNING,
            message=(
                f"{step.state}: distance={step.goal_distance:.3f}m, "
                f"front={step.scan.front:.3f}m, replans={step.replans}"
            ),
            eta=eta,
            feedback_text=(
                f"x={step.pose.x:.3f}, y={step.pose.y:.3f}, "
                f"yaw={step.pose.yaw:.3f}, path={step.target_index + 1}/"
                f"{step.path_points}, linear={step.linear:.3f}, "
                f"angular={step.angular:.3f}"
            ),
        )

    def _make_navigation_outcome_feedback(
        self,
        task_id: str,
        outcome: "NavigationOutcome",
    ) -> mobility_ai_pb2.TaskFeedback:
        if outcome.cancelled:
            status_code = common_pb2.STATUS_PREEMPTED
            message = "Navigation cancelled"
        elif outcome.ok:
            status_code = common_pb2.STATUS_SUCCESS
            message = "Navigation succeeded"
        else:
            status_code = common_pb2.STATUS_FAILURE
            message = "Navigation failed"
        return self._make_feedback(
            task_id=task_id,
            status_code=status_code,
            message=(
                f"{message}: final_distance={outcome.final_distance:.3f}m, "
                f"replans={outcome.replans}"
            ),
            feedback_text=(
                f"final_x={outcome.final_pose.x:.3f}, "
                f"final_y={outcome.final_pose.y:.3f}, "
                f"track_steps={outcome.track_steps}, "
                f"slow_steps={outcome.slow_steps}, "
                f"blocked_steps={outcome.blocked_steps}"
            ),
        )

    def _apply_controls_locked(self) -> None:
        gravity = self._gravity_compensation_locked()
        self._data.ctrl[:] = 0
        self._data.qfrc_applied[:] = 0
        for info in self._joint_infos:
            if not info.controllable:
                continue
            torque = float(gravity[info.qvel_adr])
            mode_target = self._control_targets.get(info.name)

            # If this joint has a MuJoCo actuator and the command is velocity mode,
            # prefer the native actuator path. This is important for wheel joints.
            if mode_target is not None:
                mode, target = mode_target

                if (
                    mode == core_pb2.JointCommand.ControlMode.VELOCITY
                    and info.actuator_id is not None
                    and info.actuator_control_mode == int(mode)
                ):
                    self._data.ctrl[info.actuator_id] = self._clamp_actuator_target(
                        info,
                        float(target),
                    )
                    continue

            torque = float(gravity[info.qvel_adr])

            if mode_target is not None:
                mode, target = mode_target
                qpos = float(self._data.qpos[info.qpos_adr])
                qvel = float(self._data.qvel[info.qvel_adr])
                if mode == core_pb2.JointCommand.ControlMode.POSITION:
                    if info.actuator_id is not None and info.actuator_control_mode == int(mode):
                        self._data.ctrl[info.actuator_id] = self._clamp_actuator_target(
                            info,
                            float(target),
                        )
                        torque -= POSITION_KD * qvel
                        self._data.qfrc_applied[info.qvel_adr] = torque
                        continue
                    torque += POSITION_KP * (target - qpos) - POSITION_KD * qvel
                elif mode == core_pb2.JointCommand.ControlMode.VELOCITY:
                    if self._apply_position_actuator_velocity_target_locked(
                        info,
                        qpos,
                        float(target),
                    ):
                        self._data.qfrc_applied[info.qvel_adr] = torque
                        continue
                    torque += VELOCITY_KP * (target - qvel)
                elif mode == core_pb2.JointCommand.ControlMode.TORQUE:
                    self._neutralize_position_actuator_locked(info, qpos)
                    torque += target
                else:
                    raise ValueError(f"Unsupported control mode: {mode}")
            if info.joint_actuator_forcerange is not None:
                lower, upper = info.joint_actuator_forcerange
                torque = min(max(torque, lower), upper)
            if info.actuator_id is not None:
                self._data.ctrl[info.actuator_id] = torque
            else:
                self._data.qfrc_applied[info.qvel_adr] = torque

    def _apply_position_actuator_velocity_target_locked(
        self, info: JointInfo, qpos: float, velocity: float
    ) -> bool:
        if info.actuator_id is None or info.actuator_control_mode != int(
            core_pb2.JointCommand.ControlMode.POSITION
        ):
            return False
        target = self._position_velocity_targets.get(info.name, qpos)
        target = self._clamp_actuator_target(
            info,
            target + velocity * float(self._model.opt.timestep),
        )
        self._position_velocity_targets[info.name] = target
        self._data.ctrl[info.actuator_id] = target
        return True

    def _neutralize_position_actuator_locked(self, info: JointInfo, qpos: float) -> None:
        if info.actuator_id is None or info.actuator_control_mode != int(
            core_pb2.JointCommand.ControlMode.POSITION
        ):
            return
        self._data.ctrl[info.actuator_id] = self._clamp_actuator_target(info, qpos)

    def _clamp_actuator_target(self, info: JointInfo, target: float) -> float:
        if info.actuator_ctrlrange is None:
            return target
        low, high = info.actuator_ctrlrange
        return max(low, min(high, target))

    def _gravity_compensation_locked(self) -> np.ndarray:
        self._gravity_data.qpos[:] = self._data.qpos
        self._gravity_data.qvel[:] = 0
        self._gravity_data.act[:] = self._data.act
        self._gravity_data.ctrl[:] = 0
        self._gravity_data.qfrc_applied[:] = 0
        self._gravity_data.xfrc_applied[:] = 0
        mujoco.mj_forward(self._model, self._gravity_data)
        return self._gravity_data.qfrc_bias.copy()

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

    def _set_idle_hold_targets_locked(self) -> None:
        self._position_velocity_targets.clear()
        self._control_targets = {
            info.name: (
                core_pb2.JointCommand.ControlMode.POSITION,
                float(self._data.qpos[info.qpos_adr]),
            )
            for info in self._joint_infos
            if info.controllable and self._should_idle_hold_joint(info)
        }

    def _should_idle_hold_joint(self, info: JointInfo) -> bool:
        if info.joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            return True
        if self._model.jnt_limited[info.joint_id]:
            return True
        return info.actuator_control_mode == int(core_pb2.JointCommand.ControlMode.POSITION)

    def _apply_default_configuration_locked(self) -> None:
        preferred_states = ("home", "ready", "default")
        applied = False
        for group_name in sorted(self._joint_groups):
            group = self._joint_groups[group_name]
            values = next(
                (
                    group.named_states[name]
                    for name in preferred_states
                    if name in group.named_states
                ),
                None,
            )
            if values is None:
                continue
            for joint_name, value in zip(group.joint_names, values, strict=True):
                info = self._joint_infos_by_name[joint_name]
                self._data.qpos[info.qpos_adr] = value
            applied = True
        if applied:
            self._data.qvel[:] = 0
            self._seed_position_actuator_controls_locked()
            return
        if self._apply_named_keyframe_locked(preferred_states):
            self._seed_position_actuator_controls_locked()

    def _apply_named_keyframe_locked(self, preferred_states: tuple[str, ...]) -> bool:
        for state_name in preferred_states:
            key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, state_name)
            if key_id < 0:
                continue
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
            if self._model.nu:
                self._data.ctrl[:] = self._model.key_ctrl[key_id]
            self._data.qvel[:] = 0
            return True
        return False

    def _seed_position_actuator_controls_locked(self) -> None:
        for info in self._joint_infos:
            if info.actuator_id is None or info.actuator_control_mode != int(
                core_pb2.JointCommand.ControlMode.POSITION
            ):
                continue
            self._data.ctrl[info.actuator_id] = self._clamp_actuator_target(
                info,
                float(self._data.qpos[info.qpos_adr]),
            )

    def _solve_end_effector_twist(
        self, parent_group: str, linear: np.ndarray, angular: np.ndarray
    ) -> tuple[list[str], list[float]]:
        """
        Solve for joint velocities that achieve the desired end-effector twist.
        :return: (joint names, joint velocities)
        """
        ee = self._get_end_effector(parent_group)
        group = self._joint_groups[parent_group]
        # collect all the qvel_adr of joints in the specified group
        # (to extract the corresponding columns for the Jacobian)
        dof_indices = [self._joint_infos_by_name[name].qvel_adr for name in group.joint_names]
        # linear and angular velocity Jacobians
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        with self._state_lock:
            if ee.site_name is not None:
                site_id = self._model.site(ee.site_name).id
                # compute the Jacobian for the end effector site (in world coordinate system)
                mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
            elif ee.body_name is not None:
                body_id = self._model.body(ee.body_name).id
                # compute the Jacobian for the end effector body (in world coordinate system)
                mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)
            else:
                raise ValueError(f"End effector '{ee.name}' has no MuJoCo body or site binding")
        # only keep the columns corresponding to the joints in the specified group
        jacobian = np.vstack((jacp[:, dof_indices], jacr[:, dof_indices]))
        # combine the linear and angular velocity components into a single target twist
        target_twist = np.concatenate((linear, angular))
        # Use DLS to reduce numerical issues near singular Jacobians.
        # NOTE: A fixed damping value of EE_DLS_DAMPING adds tracking error away from
        # singular configurations.
        # TODO: Adapt damping from the Jacobian singular values, for example via
        # `np.linalg.svd(jacobian)`, so it rises near singularities and falls away
        # from them.
        # But SSRVodka is lazy, so we use a fixed damping value for now. ;)
        # $J^T J + \lambda I$
        lhs = jacobian.T @ jacobian + EE_DLS_DAMPING * np.eye(len(dof_indices))
        # $J^T \cdot \tau_{desired}$
        rhs = jacobian.T @ target_twist
        # resolve $(J^T J + \lambda I) \cdot qvel = J^T \cdot \tau_{desired}$
        qvel = np.linalg.solve(lhs, rhs)
        return group.joint_names, [float(value) for value in qvel]

    # NOTE: only one end effector is supported for now
    def _get_end_effector(self, parent_group: str) -> EndEffectorInfo:
        group = self._joint_groups.get(parent_group)
        if group is None or not group.end_effectors:
            raise NotImplementedError(f"Joint model group '{parent_group}' has no end effector")
        return group.end_effectors[0]

    def _get_end_effector_pose_locked(self, ee: EndEffectorInfo) -> common_pb2.Pose:
        if ee.body_name is not None:
            body = self._data.body(ee.body_name)
            return common_pb2.Pose(
                position=self._build_point(body.xpos),
                orientation=self._build_quaternion(body.xquat),
            )
        if ee.site_name is not None:
            site = self._data.site(ee.site_name)
            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, site.xmat)
            return common_pb2.Pose(
                position=self._build_point(site.xpos),
                orientation=self._build_quaternion(quat),
            )
        raise ValueError(f"End effector '{ee.name}' has no pose binding")

    def _sensor_values(self, sensor_id: int) -> np.ndarray:
        adr = int(self._model.sensor_adr[sensor_id])
        dim = int(self._model.sensor_dim[sensor_id])
        return self._data.sensordata[adr : adr + dim].copy()

    def _render_camera_locked(self, sensor: SensorInfo) -> sensing_pb2.CameraImage:
        assert sensor.source_id is not None
        width, height = self._camera_resolution(sensor.source_id)
        renderer_key = (threading.get_ident(), width, height)
        renderer = self._renderers.get(renderer_key)
        if renderer is None:
            renderer = mujoco.Renderer(self._model, height=height, width=width)
            self._renderers[renderer_key] = renderer
        renderer.update_scene(self._data, camera=sensor.name)
        pixels = renderer.render()
        return sensing_pb2.CameraImage(
            header=self._build_header(frame_id=sensor.name),
            name=sensor.name,
            height=height,
            width=width,
            encoding="rgb8",
            is_bigendian=False,
            step=width * 3,
            data=pixels.tobytes(),
        )

    def _camera_resolution(self, camera_id: int) -> tuple[int, int]:
        raw_width = int(self._model.cam_resolution[camera_id][0])
        raw_height = int(self._model.cam_resolution[camera_id][1])
        if raw_width <= 1 and raw_height <= 1:
            return DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
        return raw_width, raw_height

    def _launch_viewer(self) -> mujoco.viewer.Handle:
        return mujoco.viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=False,
            show_right_ui=False,
        )

    def _joint_dims(self, joint_type: int) -> tuple[int, int]:
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return 7, 6
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            return 4, 3
        return 1, 1

    def _joint_type_name(self, joint_type: int) -> str:
        mapping = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        return mapping.get(joint_type, "unknown")

    def _is_robot_joint_type(self, joint_type: int) -> bool:
        return joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )

    def _parse_srdf_groups(self, root: ET.Element) -> dict[str, list[str]]:
        raw: dict[str, ET.Element] = {
            group.attrib["name"]: group for group in root.findall("group") if "name" in group.attrib
        }
        resolved: dict[str, list[str]] = {}

        def resolve(name: str) -> list[str]:
            if name in resolved:
                return resolved[name]
            group = raw[name]
            ordered: list[str] = []
            for child in group:
                if child.tag == "chain":
                    ordered.extend(
                        self._chain_joint_names(
                            child.attrib["base_link"],
                            child.attrib["tip_link"],
                        )
                    )
                elif child.tag in {"joint", "passive_joint"}:
                    ordered.append(child.attrib["name"])
                elif child.tag == "link":
                    joint_name = self._body_joint_name.get(child.attrib["name"])
                    if joint_name is not None:
                        ordered.append(joint_name)
                elif child.tag == "group":
                    ordered.extend(resolve(child.attrib["name"]))
            resolved[name] = self._dedupe_keep_order(
                [joint_name for joint_name in ordered if joint_name in self._joint_infos_by_name]
            )
            return resolved[name]

        for name in raw:
            resolve(name)
        return resolved

    def _parse_srdf_group_states(self, root: ET.Element) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for state in root.findall("group_state"):
            group_name = state.attrib.get("group")
            state_name = state.attrib.get("name")
            if not group_name or not state_name:
                continue
            joint_values = {
                joint.attrib["name"]: float(joint.attrib["value"])
                for joint in state.findall("joint")
                if "name" in joint.attrib and "value" in joint.attrib
            }
            result.setdefault(group_name, {})[state_name] = joint_values
        return result

    def _parse_srdf_end_effectors(
        self,
        root: ET.Element,
        groups: dict[str, JointModelGroup],
    ) -> list[EndEffectorInfo]:
        end_effectors: list[EndEffectorInfo] = []
        for element in root.findall("end_effector"):
            name = element.attrib.get("name")
            group_name = element.attrib.get("group")
            parent_group = element.attrib.get("parent_group")
            parent_link = element.attrib.get("parent_link", "")
            if not name or not group_name or not parent_group:
                continue
            body_name = name if self._has_body(name) else None
            site_name = name if self._has_site(name) else None
            if body_name is None and site_name is None:
                group = groups.get(group_name)
                if group is not None:
                    body_name = group.tip_body_name
            end_effectors.append(
                EndEffectorInfo(
                    name=name,
                    parent_group=parent_group,
                    group_name=group_name,
                    parent_link=parent_link,
                    body_name=body_name,
                    site_name=site_name,
                )
            )
        return end_effectors

    def _chain_joint_names(self, base_link: str, tip_link: str) -> list[str]:
        if not self._has_body(base_link) or not self._has_body(tip_link):
            return []
        current_id = self._model.body(tip_link).id
        base_id = self._model.body(base_link).id
        chain: list[str] = []
        while current_id != base_id:
            body_name = self._model.body(current_id).name
            joint_name = self._body_joint_name.get(body_name)
            if joint_name is not None:
                chain.append(joint_name)
            current_id = int(self._model.body_parentid[current_id])
            if current_id <= 0:
                break
        chain.reverse()
        return chain

    def _infer_group_tip_body(self, joint_names: list[str]) -> str | None:
        if not joint_names:
            return None
        body_ids = [self._joint_infos_by_name[name].body_id for name in joint_names]
        depths = {body_id: self._body_depth(body_id) for body_id in body_ids}
        return self._model.body(max(body_ids, key=lambda body_id: depths[body_id])).name

    def _body_depth(self, body_id: int) -> int:
        depth = 0
        current = body_id
        while current > 0:
            current = int(self._model.body_parentid[current])
            depth += 1
        return depth

    def _has_body(self, name: str) -> bool:
        try:
            self._model.body(name)
            return True
        except KeyError:
            return False

    def _has_site(self, name: str) -> bool:
        try:
            self._model.site(name)
            return True
        except KeyError:
            return False

    def _map_sensor_type(self, sensor_type: int) -> SensorType | None:
        mapping = {
            int(mujoco.mjtSensor.mjSENS_FORCE): SensorType.FORCE,
            int(mujoco.mjtSensor.mjSENS_TORQUE): SensorType.TORQUE,
            int(mujoco.mjtSensor.mjSENS_ACCELEROMETER): SensorType.IMU,
            int(mujoco.mjtSensor.mjSENS_GYRO): SensorType.IMU,
        }
        return mapping.get(sensor_type)

    def _build_header(self, frame_id: str) -> common_pb2.Header:
        return common_pb2.Header(
            seq=0,
            timestamp=float(self._data.time),
            frame_id=frame_id,
        )

    def _build_point(self, values: np.ndarray | list[float]) -> common_pb2.Point:
        return common_pb2.Point(
            x=float(values[0]) if len(values) > 0 else 0.0,
            y=float(values[1]) if len(values) > 1 else 0.0,
            z=float(values[2]) if len(values) > 2 else 0.0,
        )

    def _build_quaternion(self, values: np.ndarray | list[float]) -> common_pb2.Quaternion:
        return common_pb2.Quaternion(
            x=float(values[1]),
            y=float(values[2]),
            z=float(values[3]),
            w=float(values[0]),
        )

    def _dedupe_keep_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
