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
        self._robot_body_ids = self._collect_subtree_body_ids(self._robot_root_body_id)
        self._joint_infos = self._build_joint_infos()
        self._joint_infos_by_name = {info.name: info for info in self._joint_infos}
        self._body_children = self._build_body_children()
        self._body_joint_name = self._build_body_joint_name_map()
        self._controllable_joint_names = [
            info.name for info in self._joint_infos if info.controllable
        ]
        self._control_targets: dict[str, tuple[core_pb2.JointCommand.ControlMode, float]] = {}
        self._joint_groups = self._build_joint_groups()
        self._joint_to_groups = self._build_joint_to_groups()
        self._sensors = self._build_sensors()
        self._capabilities = self._detect_capabilities()

        self._viewer: mujoco.viewer.Handle | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
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

    def _collect_subtree_body_ids(self, body_id: int) -> set[int]:
        result = {body_id}
        for child_id in range(1, self._model.nbody):
            if int(self._model.body_parentid[child_id]) == body_id:
                result.update(self._collect_subtree_body_ids(child_id))
        return result

    def _build_body_children(self) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {
            body_id: [] for body_id in range(self._model.nbody)
        }
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
        actuator_by_joint_id: dict[int, tuple[int, tuple[float, float]]] = {}
        for actuator_id in range(self._model.nu):
            trn_type = int(self._model.actuator_trntype[actuator_id])
            if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            joint_id = int(self._model.actuator_trnid[actuator_id][0])
            ctrlrange = self._model.actuator_ctrlrange[actuator_id]
            actuator_by_joint_id[joint_id] = (
                actuator_id,
                (float(ctrlrange[0]), float(ctrlrange[1])),
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
                )
            )
        return joint_infos

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
                    core_pb2.JointCommand.ControlMode(joint_cmd.mode),
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
                    if sensor.sensor_type == SensorType.FORCE:
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
            # No compensation: self._control_targets.clear()
            self._paused = False
            self._apply_default_configuration_locked()
            mujoco.mj_forward(self._model, self._data)
            self._set_idle_hold_targets_locked()
            self._sync_viewer_locked()

    def emergency_stop(self) -> None:
        with self._state_lock:
            self._control_targets.clear()
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

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        raise NotImplementedError("Navigation not supported for MuJoCo")

    def navigate_to(self, goal: mobility_ai_pb2.NavGoal) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        raise NotImplementedError("Navigation not supported for MuJoCo")

    def _apply_controls_locked(self) -> None:
        gravity = self._gravity_compensation_locked()
        self._data.ctrl[:] = 0
        self._data.qfrc_applied[:] = 0
        for info in self._joint_infos:
            if not info.controllable:
                continue
            torque = float(gravity[info.qvel_adr])
            mode_target = self._control_targets.get(info.name)
            if mode_target is not None:
                mode, target = mode_target
                qpos = float(self._data.qpos[info.qpos_adr])
                qvel = float(self._data.qvel[info.qvel_adr])
                if mode == core_pb2.JointCommand.ControlMode.POSITION:
                    torque += POSITION_KP * (target - qpos) - POSITION_KD * qvel
                elif mode == core_pb2.JointCommand.ControlMode.VELOCITY:
                    torque += VELOCITY_KP * (target - qvel)
                elif mode == core_pb2.JointCommand.ControlMode.TORQUE:
                    torque += target
                else:
                    raise ValueError(f"Unsupported control mode: {mode}")
            self._data.qfrc_applied[info.qvel_adr] = torque

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
        self._control_targets = {
            info.name: (
                core_pb2.JointCommand.ControlMode.POSITION,
                float(self._data.qpos[info.qpos_adr]),
            )
            for info in self._joint_infos
            if info.controllable
        }

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
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = mujoco.Renderer(self._model, height=height, width=width)
            self._renderers[(width, height)] = renderer
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

    def _parse_srdf_group_states(
        self, root: ET.Element
    ) -> dict[str, dict[str, dict[str, float]]]:
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
