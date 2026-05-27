"""Habitat-Sim render-only backend implementation."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from control_stubs.sensing_pb2 import SensorType

from control_stubs import common_pb2, mobility_ai_pb2, sensing_pb2
from control_stubs import robot_core_pb2 as core_pb2
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability

DEFAULT_CAMERA_NAME = "habitat_rgb"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_POSITION = (0.0, 1.5, 0.0)
PANDA_CAMERA_POSITION = (0.0, 1.0, 2.2)
STREAM_INTERVAL_SEC = 0.05
SERVO_DT_SEC = 0.05
EE_DLS_DAMPING = 1e-4
EE_FD_STEP = 1e-4
EE_MAX_JOINT_VELOCITY = 1.0
SUPPORTED_VIEWER_SCENE_SUFFIXES = {".glb", ".gltf", ".obj", ".ply"}
DRIVERS_SIM_ROOT_ENV = "ROBOSIM_DRIVERS_SIM_ROOT"
PANDA_EE_NAME = "hand"
PANDA_EE_PARENT_GROUP = "panda_arm"
PANDA_EE_JOINT_NAME = "panda_joint7"
CAMERA_OBJECT_NAMES = {DEFAULT_CAMERA_NAME, "camera", "habitat_camera"}
PANDA_DEFAULT_JOINT_POSITIONS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
    "panda_finger_joint1": 0.02,
    "panda_finger_joint2": 0.02,
}
PANDA_ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
PANDA_HAND_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


@dataclass(slots=True)
class CameraConfig:
    name: str = DEFAULT_CAMERA_NAME
    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    position: tuple[float, float, float] = DEFAULT_CAMERA_POSITION


@dataclass(slots=True)
class HabitatJointInfo:
    name: str
    link_id: int
    position_index: int
    type: str
    lower_limit: float
    upper_limit: float


class HabitatSimBackend(SimulatorBackend):
    """Render-only Habitat-Sim backend.

    Habitat-Sim does not model the RoboSim robot/control contract here. This
    backend intentionally exposes camera sensing and simulation reset only.
    """

    def __init__(
        self,
        scene_path: str | None = None,
        headless: bool = True,
        robot: str | None = None,
        enable_camera: bool | None = None,
        camera: CameraConfig | None = None,
    ) -> None:
        self._habitat_sim = self._import_habitat_sim()
        self._scene_path = self._resolve_scene_path(scene_path)
        self._headless_mode = headless
        self._requested_robot = Path(robot).expanduser() if robot is not None else None
        self._resolved_robot_urdf: Path | None = None
        self._resolved_robot_name: str | None = None
        self._enable_camera_override = enable_camera
        self._camera = camera or self._default_camera()
        self._camera_world_position = tuple(self._camera.position)
        self._camera_world_orientation = (0.0, 0.0, 0.0, 1.0)
        self._seq = 0
        self._closed = False
        self._viewer: subprocess.Popen[bytes] | None = None
        self._robot: Any | None = None
        self._robot_urdf_path: Path | None = None
        self._joint_infos: list[HabitatJointInfo] = []
        self._sim = None if self._uses_display_viewer else self._create_simulator()
        if self._sim is not None:
            self._robot = self._load_robot()
        if self._uses_display_viewer:
            if self._uses_robot:
                raise NotImplementedError(
                    "Habitat-Sim robot support uses the Simulator API and is not "
                    "available through the display viewer subprocess. Use --headless."
                )
            self._viewer = self._launch_display_viewer()

    @property
    def capabilities(self) -> Capability:
        capabilities = Capability.SIMULATION_CONTROL
        if self._camera_enabled:
            capabilities |= Capability.SENSOR_CAMERA
        if self._robot is not None:
            capabilities |= Capability.JOINT_READ | Capability.JOINT_WRITE | Capability.SENSOR_JOINT
        return capabilities

    @property
    def robot_name(self) -> str:
        if self._uses_robot:
            return self._resolved_robot_name or self._robot_name_from_path()
        return "habitat_camera"

    @property
    def headless_mode(self) -> bool:
        return self._headless_mode

    def set_headless_mode(self, enabled: bool) -> None:
        self._headless_mode = enabled

    def get_robot_state(self) -> common_pb2.JointState:
        if self._robot is None:
            return common_pb2.JointState(header=self._build_header(frame_id="world"))
        positions = list(self._robot.joint_positions)
        return common_pb2.JointState(
            header=self._build_header(frame_id="world"),
            name=[info.name for info in self._joint_infos],
            position=[positions[info.position_index] for info in self._joint_infos],
            velocity=[0.0 for _ in self._joint_infos],
            effort=[0.0 for _ in self._joint_infos],
        )

    def get_joint_command_state(self) -> common_pb2.JointState:
        return self.get_robot_state()

    def get_robot_spec(self) -> core_pb2.RobotSpecification:
        spec = core_pb2.RobotSpecification(robot_name=self.robot_name)
        for info in self._joint_infos:
            spec.joints.append(
                core_pb2.JointLimit(
                    name=info.name,
                    type=info.type,
                    jmg_names=self._joint_groups_for_name(info.name),
                    lower_limit=info.lower_limit,
                    upper_limit=info.upper_limit,
                )
            )
        if self._uses_panda_robot:
            ready = [PANDA_DEFAULT_JOINT_POSITIONS[name] for name in PANDA_ARM_JOINTS]
            spec.joint_model_groups.extend(
                [
                    core_pb2.JointModelGroupSpec(
                        name="panda_arm",
                        joint_names=PANDA_ARM_JOINTS,
                        named_states=[
                            core_pb2.JointModelGroupNamedState(
                                name="ready",
                                joint_values=ready,
                            )
                        ],
                        end_effectors=[
                            core_pb2.EESpec(
                                name=PANDA_EE_NAME,
                                parent_jmg_name=PANDA_EE_PARENT_GROUP,
                                parent_link=PANDA_EE_JOINT_NAME,
                            )
                        ],
                    ),
                    core_pb2.JointModelGroupSpec(
                        name="panda_hand",
                        joint_names=PANDA_HAND_JOINTS,
                    ),
                    core_pb2.JointModelGroupSpec(
                        name="panda_arm_hand",
                        joint_names=PANDA_ARM_JOINTS + PANDA_HAND_JOINTS,
                    ),
                ]
            )
        return spec

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: core_pb2.JointCommand.ControlMode,
        group: str | None = None,
    ) -> None:
        del group
        if self._robot is None:
            raise NotImplementedError("Robot joint control is not supported for Habitat-Sim")
        if mode not in (
            core_pb2.JointCommand.POSITION,
            core_pb2.JointCommand.VELOCITY,
        ):
            raise NotImplementedError(
                "Habitat-Sim Panda support currently accepts POSITION or VELOCITY commands only"
            )
        if len(names) != len(data):
            raise ValueError("Joint command names and data must have the same length")

        positions = list(self._robot.joint_positions)
        joints_by_name = {info.name: info for info in self._joint_infos}
        for name, value in zip(names, data, strict=True):
            info = joints_by_name.get(name)
            if info is None:
                raise ValueError(f"Unknown Habitat-Sim joint '{name}'")
            if mode == core_pb2.JointCommand.POSITION:
                target = value
            else:
                target = positions[info.position_index] + value * SERVO_DT_SEC
            positions[info.position_index] = min(max(target, info.lower_limit), info.upper_limit)
        self._robot.joint_positions = positions

    def servo_control_stream(
        self,
        request_iterator: Iterator[core_pb2.ServoCommand],
    ) -> Iterator[common_pb2.JointState]:
        if self._robot is None:
            raise NotImplementedError("Robot servo control is not supported for Habitat-Sim")
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
                if parent_group != PANDA_EE_PARENT_GROUP:
                    raise ValueError(
                        f"Habitat-Sim Panda supports twist servo only for {PANDA_EE_PARENT_GROUP!r}"
                    )
                joint_names, joint_velocities = self._solve_end_effector_twist(
                    linear=[
                        twist_cmd.twist.twist.linear.x,
                        twist_cmd.twist.twist.linear.y,
                        twist_cmd.twist.twist.linear.z,
                    ],
                    angular=[
                        twist_cmd.twist.twist.angular.x,
                        twist_cmd.twist.twist.angular.y,
                        twist_cmd.twist.twist.angular.z,
                    ],
                )
                self.set_joint_target(
                    joint_names,
                    joint_velocities,
                    core_pb2.JointCommand.VELOCITY,
                    PANDA_EE_PARENT_GROUP,
                )
            yield self.get_robot_state()

    def get_end_effector_state(self, group: str) -> core_pb2.EndEffectorState:
        if self._robot is None or group != PANDA_EE_PARENT_GROUP:
            raise NotImplementedError(
                f"Habitat-Sim Panda exposes only the {PANDA_EE_PARENT_GROUP!r} end effector"
            )
        return core_pb2.EndEffectorState(
            pose_stamped=common_pb2.PoseStamped(
                header=self._build_header(frame_id="world"),
                pose=self._get_end_effector_pose(),
            )
        )

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        if not self._camera_enabled:
            return sensing_pb2.SensorMetaList()
        return sensing_pb2.SensorMetaList(
            entries=[
                sensing_pb2.SensorMetaList.SensorMeta(
                    name=self._camera.name,
                    type=SensorType.CAMERA,
                )
            ]
        )

    def get_sensors(self, names: list[str]) -> sensing_pb2.SensorData:
        if not self._camera_enabled:
            raise NotImplementedError(
                "Habitat-Sim Panda mode disables camera rendering by default so it "
                "can run without an EGL/GPU context. Pass enable_camera=True, or "
                "use --habitat-enable-camera from the server CLI, on a GPU machine."
            )
        if self._uses_display_viewer:
            raise NotImplementedError(
                "Habitat-Sim display viewer mode renders to a local window and "
                "does not expose camera images through gRPC. Use --headless on "
                "a machine with working EGL to stream habitat_rgb."
            )
        requested = set(names) if names else {self._camera.name}
        if self._camera.name not in requested:
            return sensing_pb2.SensorData()
        return sensing_pb2.SensorData(images=[self._render_camera()])

    def stream_sensors(self, names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        while not self._closed:
            yield self.get_sensors(names)
            time.sleep(STREAM_INTERVAL_SEC)

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        raise NotImplementedError("Navigation is not supported for Habitat-Sim")

    def navigate_to(self, goal: mobility_ai_pb2.NavGoal) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        del goal
        raise NotImplementedError("Navigation is not supported for Habitat-Sim")

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params
        if self._sim is not None and hasattr(self._sim, "reset"):
            self._sim.reset()
            if self._camera_enabled:
                self._apply_camera_pose()

    def set_object_pose(self, object_name: str, pose: common_pb2.Pose) -> None:
        if object_name not in CAMERA_OBJECT_NAMES:
            raise NotImplementedError(
                "Habitat-Sim currently supports SetObjectPose only for "
                f"{sorted(CAMERA_OBJECT_NAMES)}"
            )
        self._camera_world_position = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )
        self._camera_world_orientation = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self._apply_camera_pose()

    def emergency_stop(self) -> None:
        return None

    def shutdown(self) -> None:
        self._closed = True
        if self._viewer is not None:
            self._viewer.terminate()
            try:
                self._viewer.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._viewer.kill()
            self._viewer = None
        if self._sim is not None and hasattr(self._sim, "close"):
            self._sim.close()
        if self._robot_urdf_path is not None:
            self._robot_urdf_path.unlink(missing_ok=True)
            self._robot_urdf_path = None

    @property
    def _uses_display_viewer(self) -> bool:
        return not self._headless_mode

    def _launch_display_viewer(self) -> subprocess.Popen[bytes]:
        self._validate_display_scene()
        viewer = shutil.which("viewer")
        if viewer is None:
            candidate = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "viewer"
            if candidate.exists():
                viewer = str(candidate)
        if viewer is None:
            raise RuntimeError("Habitat-Sim display viewer executable not found")

        env = os.environ.copy()
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        env.setdefault("MESA_GL_VERSION_OVERRIDE", "4.1")
        if not env.get("DISPLAY"):
            env["DISPLAY"] = ":0"

        return subprocess.Popen(
            [viewer, self._scene_path],
            env=env,
        )

    def _validate_display_scene(self) -> None:
        scene_path = Path(self._scene_path)
        if self._scene_path == "NONE":
            raise ValueError(
                "Habitat-Sim display viewer requires a mesh scene. "
                "Pass --scene with a .glb, .gltf, .obj, or .ply file."
            )
        if scene_path.suffix.lower() == ".xml" and self._looks_like_mjcf(scene_path):
            raise ValueError(
                "Habitat-Sim viewer cannot load MuJoCo MJCF XML scenes. "
                "Use a mesh scene such as "
                "drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/BEDROOM_NEO/model.obj"
            )
        if scene_path.suffix.lower() not in SUPPORTED_VIEWER_SCENE_SUFFIXES:
            raise ValueError(
                "Habitat-Sim display viewer expects one of "
                f"{sorted(SUPPORTED_VIEWER_SCENE_SUFFIXES)}, got '{scene_path.suffix}'."
            )

    def _looks_like_mjcf(self, scene_path: Path) -> bool:
        if not scene_path.exists():
            return False
        try:
            return ET.parse(scene_path).getroot().tag == "mujoco"
        except ET.ParseError:
            return False

    def _create_simulator(self) -> Any:
        habitat_sim = self._habitat_sim
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self._scene_path
        if hasattr(sim_cfg, "create_renderer"):
            sim_cfg.create_renderer = self._camera_enabled
        if hasattr(sim_cfg, "enable_physics"):
            sim_cfg.enable_physics = self._uses_robot

        agent_cfg = habitat_sim.AgentConfiguration()
        if self._camera_enabled:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = self._camera.name
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [self._camera.height, self._camera.width]
            sensor_spec.position = [0.0, 0.0, 0.0]
            agent_cfg.sensor_specifications = [sensor_spec]

        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        if self._camera_enabled:
            self._apply_camera_pose(sim)
        return sim

    @property
    def _camera_enabled(self) -> bool:
        if self._enable_camera_override is not None:
            return self._enable_camera_override
        return not self._uses_robot

    @property
    def _uses_panda_robot(self) -> bool:
        return self._resolved_robot_name == "panda" or self._robot_name_from_path() == "panda"

    @property
    def _uses_robot(self) -> bool:
        return self._requested_robot is not None

    def _default_camera(self) -> CameraConfig:
        if self._uses_robot:
            return CameraConfig(position=PANDA_CAMERA_POSITION)
        return CameraConfig()

    def _apply_camera_pose(self, sim: Any | None = None) -> None:
        sim = sim if sim is not None else self._sim
        if sim is None:
            return
        agent = self._get_agent(sim)
        if agent is None:
            return

        habitat_sim = self._habitat_sim
        state_cls = getattr(habitat_sim, "AgentState", None)
        state = state_cls() if state_cls is not None else self._current_agent_state(agent)
        if state is None:
            return

        state.position = list(self._camera_world_position)
        rotation = self._habitat_quaternion(self._camera_world_orientation)
        if rotation is not None:
            state.rotation = rotation
        else:
            state.rotation = self._camera_world_orientation

        if hasattr(agent, "set_state"):
            agent.set_state(state)
        elif hasattr(sim, "set_agent_state"):
            sim.set_agent_state(state.position, state.rotation)

    def _get_agent(self, sim: Any) -> Any | None:
        if hasattr(sim, "get_agent"):
            return sim.get_agent(0)
        if hasattr(sim, "initialize_agent"):
            return sim.initialize_agent(0)
        return None

    def _current_agent_state(self, agent: Any) -> Any | None:
        state = getattr(agent, "state", None)
        if state is not None:
            return state
        get_state = getattr(agent, "get_state", None)
        if callable(get_state):
            return get_state()
        return None

    def _habitat_quaternion(self, xyzw: tuple[float, float, float, float]) -> Any | None:
        try:
            from habitat_sim.utils.common import quat_from_coeffs
        except ImportError:
            return None
        return quat_from_coeffs(list(xyzw))

    def _load_robot(self) -> Any | None:
        if not self._uses_robot:
            return None
        if self._sim is None:
            return None
        robot = self._sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            str(self._prepare_robot_urdf()),
            fixed_base=True,
            maintain_link_order=True,
        )
        self._joint_infos = self._build_robot_joint_infos(robot)
        if self._uses_panda_robot:
            self._apply_panda_ready_state(robot)
        return robot

    def _prepare_robot_urdf(self) -> Path:
        source = self._resolve_robot_urdf_path()
        if not source.exists():
            raise FileNotFoundError(f"Could not find Habitat-Sim robot URDF: {source}")
        self._resolved_robot_urdf = source
        self._resolved_robot_name = self._read_urdf_robot_name(source) or source.stem
        mesh_root = source.parents[1]
        text = source.read_text()
        text = self._resolve_package_mesh_uris(text)
        text = text.replace("package://franka_panda_desc/", f"{mesh_root}/")
        handle = tempfile.NamedTemporaryFile(
            "w",
            suffix=".urdf",
            prefix="robosim_habitat_panda_",
            delete=False,
        )
        with handle:
            handle.write(text)
        self._robot_urdf_path = Path(handle.name)
        return self._robot_urdf_path

    def _resolve_robot_urdf_path(self) -> Path:
        if self._requested_robot is None:
            raise ValueError("Habitat-Sim robot loading requires --robot")

        robot_path = self._resolve_requested_robot_path()

        if robot_path.is_file():
            if robot_path.suffix.lower() != ".urdf":
                raise ValueError(
                    "Habitat-Sim articulated robot loading expects a URDF file. "
                    f"Got {robot_path}."
                )
            return robot_path
        if not robot_path.exists():
            raise FileNotFoundError(f"Robot path does not exist: {robot_path}")
        if not robot_path.is_dir():
            raise ValueError(f"Robot path must be a directory or URDF file: {robot_path}")

        preferred = [
            robot_path / "urdf" / f"{robot_path.name}.urdf",
            robot_path / f"{robot_path.name}.urdf",
            robot_path / "robot.urdf",
            robot_path / "model.urdf",
        ]
        for candidate in preferred:
            if candidate.exists():
                return candidate

        candidates = sorted(robot_path.rglob("*.urdf"))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            names = ", ".join(str(path.relative_to(robot_path)) for path in candidates)
            raise ValueError(
                f"Multiple URDF files found in {robot_path}: {names}. "
                "Pass --robot with the specific .urdf file."
            )

        mjcf_candidates = sorted(
            path
            for path in robot_path.glob("*.xml")
            if path.name != "package.xml" and self._looks_like_mjcf(path)
        )
        if mjcf_candidates:
            raise ValueError(
                "Habitat-Sim cannot load MuJoCo MJCF XML robot files through the "
                "current articulated-object path. Provide a robot directory containing "
                "a .urdf file, or use --backend mujoco for this robot."
            )
        raise FileNotFoundError(f"No URDF file found in robot directory: {robot_path}")

    def _read_urdf_robot_name(self, urdf_path: Path) -> str | None:
        try:
            return ET.parse(urdf_path).getroot().attrib.get("name")
        except ET.ParseError:
            return None

    def _resolve_package_mesh_uris(self, text: str) -> str:
        if self._requested_robot is None:
            return text
        robot_path = self._resolve_requested_robot_path()
        robot_root = robot_path if robot_path.is_dir() else robot_path.parent
        for package_xml in robot_root.rglob("package.xml"):
            try:
                package_name = ET.parse(package_xml).getroot().findtext("name")
            except ET.ParseError:
                continue
            if package_name:
                text = text.replace(f"package://{package_name}/", f"{package_xml.parent}/")
        return text

    def _robot_name_from_path(self) -> str:
        if self._requested_robot is None:
            return "habitat_camera"
        return self._requested_robot.stem if self._requested_robot.is_file() else self._requested_robot.name

    def _resolve_requested_robot_path(self) -> Path:
        if self._requested_robot is None:
            raise ValueError("Habitat-Sim robot loading requires --robot")
        robot_path = self._requested_robot
        if robot_path.is_absolute() or robot_path.exists():
            return robot_path.resolve()

        env_candidate = self._drivers_sim_root() / robot_path
        if env_candidate.exists():
            return env_candidate.resolve()

        return (Path.cwd() / robot_path).resolve()

    def _drivers_sim_root(self) -> Path:
        env_root = os.environ.get(DRIVERS_SIM_ROOT_ENV)
        if env_root:
            return Path(env_root).expanduser().resolve()

        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "drivers_sim",
            repo_root.parent / "drivers_sim",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _build_robot_joint_infos(self, robot: Any) -> list[HabitatJointInfo]:
        lower_limits, upper_limits = robot.joint_position_limits
        infos: list[HabitatJointInfo] = []
        for link_id in robot.get_link_ids():
            num_positions = int(robot.get_link_num_joint_pos(link_id))
            if num_positions != 1:
                continue
            offset = int(robot.get_link_joint_pos_offset(link_id))
            name = str(robot.get_link_joint_name(link_id))
            infos.append(
                HabitatJointInfo(
                    name=name,
                    link_id=int(link_id),
                    position_index=offset,
                    type=self._habitat_joint_type_name(robot.get_link_joint_type(link_id)),
                    lower_limit=float(lower_limits[offset]),
                    upper_limit=float(upper_limits[offset]),
                )
            )
        return infos

    def _apply_panda_ready_state(self, robot: Any) -> None:
        positions = list(robot.joint_positions)
        for info in self._joint_infos:
            if info.name in PANDA_DEFAULT_JOINT_POSITIONS:
                positions[info.position_index] = PANDA_DEFAULT_JOINT_POSITIONS[info.name]
        robot.joint_positions = positions

    def _habitat_joint_type_name(self, joint_type: Any) -> str:
        name = getattr(joint_type, "name", str(joint_type))
        return name.lower()

    def _joint_groups_for_name(self, name: str) -> list[str]:
        groups = []
        if name in PANDA_ARM_JOINTS:
            groups.extend(["panda_arm", "panda_arm_hand"])
        if name in PANDA_HAND_JOINTS:
            groups.extend(["panda_hand", "panda_arm_hand"])
        return groups

    def _solve_end_effector_twist(
        self,
        linear: list[float],
        angular: list[float],
    ) -> tuple[list[str], list[float]]:
        import numpy as np

        if self._robot is None:
            raise RuntimeError("Habitat-Sim robot is not available")

        target_linear = np.asarray(linear, dtype=np.float64)
        target_angular = np.asarray(angular, dtype=np.float64)
        joint_infos = [info for info in self._joint_infos if info.name in PANDA_ARM_JOINTS]
        if not joint_infos:
            raise RuntimeError("Panda arm joints are not available")

        original_positions = list(self._robot.joint_positions)
        jacobian = np.zeros((3, len(joint_infos)), dtype=np.float64)
        try:
            for column, info in enumerate(joint_infos):
                center = original_positions[info.position_index]
                plus = min(center + EE_FD_STEP, info.upper_limit)
                minus = max(center - EE_FD_STEP, info.lower_limit)
                if plus == minus:
                    continue

                positions = list(original_positions)
                positions[info.position_index] = plus
                self._robot.joint_positions = positions
                plus_pos = np.asarray(self._get_end_effector_position(), dtype=np.float64)

                positions = list(original_positions)
                positions[info.position_index] = minus
                self._robot.joint_positions = positions
                minus_pos = np.asarray(self._get_end_effector_position(), dtype=np.float64)

                jacobian[:, column] = (plus_pos - minus_pos) / (plus - minus)
        finally:
            self._robot.joint_positions = original_positions

        if np.linalg.norm(target_linear) == 0.0 and np.linalg.norm(target_angular) > 0.0:
            return (
                [info.name for info in joint_infos],
                self._angular_fallback_joint_velocities(joint_infos, target_angular),
            )

        lhs = jacobian.T @ jacobian + EE_DLS_DAMPING * np.eye(len(joint_infos))
        rhs = jacobian.T @ target_linear
        velocities = np.linalg.solve(lhs, rhs)
        velocities = np.clip(velocities, -EE_MAX_JOINT_VELOCITY, EE_MAX_JOINT_VELOCITY)
        return [info.name for info in joint_infos], [float(value) for value in velocities]

    def _angular_fallback_joint_velocities(
        self,
        joint_infos: list[HabitatJointInfo],
        angular: Any,
    ) -> list[float]:
        import numpy as np

        velocities = np.zeros(len(joint_infos), dtype=np.float64)
        joint_index_by_name = {info.name: index for index, info in enumerate(joint_infos)}
        fallback_map = {
            "panda_joint5": float(angular[0]),
            "panda_joint6": float(angular[1]),
            "panda_joint7": float(angular[2]),
        }
        for name, velocity in fallback_map.items():
            index = joint_index_by_name.get(name)
            if index is not None:
                velocities[index] = velocity
        return [
            float(value)
            for value in np.clip(velocities, -EE_MAX_JOINT_VELOCITY, EE_MAX_JOINT_VELOCITY)
        ]

    def _get_end_effector_pose(self) -> common_pb2.Pose:
        link_id = self._end_effector_link_id()
        node = self._robot.get_link_scene_node(link_id)
        transform = self._node_absolute_transform(node)
        position = self._transform_translation(transform)
        orientation = self._transform_quaternion(transform)
        return common_pb2.Pose(
            position=common_pb2.Point(x=position[0], y=position[1], z=position[2]),
            orientation=common_pb2.Quaternion(
                x=orientation[0],
                y=orientation[1],
                z=orientation[2],
                w=orientation[3],
            ),
        )

    def _get_end_effector_position(self) -> tuple[float, float, float]:
        pose = self._get_end_effector_pose()
        return pose.position.x, pose.position.y, pose.position.z

    def _end_effector_link_id(self) -> int:
        for info in self._joint_infos:
            if info.name == PANDA_EE_JOINT_NAME:
                return info.link_id
        for info in reversed(self._joint_infos):
            if info.name in PANDA_ARM_JOINTS:
                return info.link_id
        raise RuntimeError("Could not resolve Panda end-effector link")

    def _node_absolute_transform(self, node: Any) -> Any:
        transform = getattr(node, "absolute_transformation", None)
        if callable(transform):
            return transform()
        if transform is not None:
            return transform
        transform = getattr(node, "transformation", None)
        if callable(transform):
            return transform()
        if transform is not None:
            return transform
        return node

    def _transform_translation(self, transform: Any) -> tuple[float, float, float]:
        translation = getattr(transform, "translation", None)
        if callable(translation):
            translation = translation()
        if translation is not None:
            return self._vector3_tuple(translation)
        if hasattr(transform, "__getitem__"):
            try:
                return float(transform[0][3]), float(transform[1][3]), float(transform[2][3])
            except (TypeError, IndexError, KeyError):
                pass
        return 0.0, 0.0, 0.0

    def _transform_quaternion(self, transform: Any) -> tuple[float, float, float, float]:
        rotation = getattr(transform, "rotation", None)
        if callable(rotation):
            rotation = rotation()
        quaternion = getattr(rotation, "to_quaternion", None) if rotation is not None else None
        if callable(quaternion):
            return self._quaternion_tuple(quaternion())
        quaternion = getattr(transform, "rotation", None)
        if quaternion is not None and not callable(quaternion):
            return self._quaternion_tuple(quaternion)
        return 0.0, 0.0, 0.0, 1.0

    def _vector3_tuple(self, value: Any) -> tuple[float, float, float]:
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            coords = []
            for axis in ("x", "y", "z"):
                coord = getattr(value, axis)
                coords.append(float(coord() if callable(coord) else coord))
            return coords[0], coords[1], coords[2]
        return float(value[0]), float(value[1]), float(value[2])

    def _quaternion_tuple(self, value: Any) -> tuple[float, float, float, float]:
        vector = getattr(value, "vector", None)
        scalar = getattr(value, "scalar", None)
        if callable(vector):
            vector = vector()
        if callable(scalar):
            scalar = scalar()
        if vector is not None and scalar is not None:
            xyz = self._vector3_tuple(vector)
            return xyz[0], xyz[1], xyz[2], float(scalar)
        if all(hasattr(value, field) for field in ("x", "y", "z", "w")):
            coords = []
            for field in ("x", "y", "z", "w"):
                coord = getattr(value, field)
                coords.append(float(coord() if callable(coord) else coord))
            return coords[0], coords[1], coords[2], coords[3]
        try:
            if len(value) == 4:
                return float(value[0]), float(value[1]), float(value[2]), float(value[3])
        except TypeError:
            pass
        return 0.0, 0.0, 0.0, 1.0

    def _render_camera(self) -> sensing_pb2.CameraImage:
        if self._sim is None:
            raise RuntimeError("Habitat-Sim simulator is not available")
        observations = self._sim.get_sensor_observations()
        if self._camera.name not in observations:
            raise RuntimeError(f"Habitat-Sim did not return sensor '{self._camera.name}'")
        pixels = self._normalize_rgb(observations[self._camera.name])
        height, width = pixels.shape[:2]
        return sensing_pb2.CameraImage(
            header=self._build_header(frame_id=self._camera.name),
            name=self._camera.name,
            height=height,
            width=width,
            encoding="rgb8",
            is_bigendian=False,
            step=width * 3,
            data=pixels.tobytes(),
        )

    def _normalize_rgb(self, image: Any) -> Any:
        import numpy as np

        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise RuntimeError(f"Expected Habitat-Sim RGB/RGBA image, got shape {pixels.shape}")
        pixels = pixels[:, :, :3]
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(pixels)

    def _build_header(self, frame_id: str) -> common_pb2.Header:
        self._seq += 1
        return common_pb2.Header(seq=self._seq, timestamp=time.time(), frame_id=frame_id)

    def _resolve_scene_path(self, scene_path: str | None) -> str:
        if scene_path is None:
            return "NONE"
        path = Path(scene_path)
        if path.exists():
            return str(path.resolve())

        drivers_root = self._drivers_sim_root()
        if path.parts and path.parts[0] == "drivers_sim":
            drivers_candidate = drivers_root.parent / path
        else:
            drivers_candidate = drivers_root / path
        if drivers_candidate.exists():
            return str(drivers_candidate.resolve())

        return scene_path

    def _import_habitat_sim(self) -> Any:
        try:
            import habitat_sim
        except ImportError as exc:
            raise ImportError(
                "Habitat-Sim backend requires the optional 'habitat_sim' package. "
                "Install Habitat-Sim in the active environment before selecting "
                "--backend habitat."
            ) from exc
        return habitat_sim
