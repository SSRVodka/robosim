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

from control_stubs import common_pb2, mobility_ai_pb2, sensing_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.sensing_pb2 import SensorType
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability

DEFAULT_CAMERA_NAME = "habitat_rgb"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_POSITION = (0.0, 1.5, 0.0)
PANDA_CAMERA_POSITION = (0.0, 1.0, 2.2)
STREAM_INTERVAL_SEC = 0.05
SUPPORTED_VIEWER_SCENE_SUFFIXES = {".glb", ".gltf", ".obj", ".ply"}
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
        robot_name: str | None = None,
        enable_camera: bool | None = None,
        camera: CameraConfig | None = None,
    ) -> None:
        self._habitat_sim = self._import_habitat_sim()
        self._scene_path = self._resolve_scene_path(scene_path)
        self._headless_mode = headless
        self._requested_robot_name = robot_name
        self._enable_camera_override = enable_camera
        self._camera = camera or self._default_camera()
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
            if self._uses_panda_robot:
                raise NotImplementedError(
                    "Habitat-Sim Panda support uses the Simulator API and is not "
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
        if self._uses_panda_robot:
            return "panda"
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
                        end_effectors=[core_pb2.EESpec(name="hand", parent_jmg_name="panda_arm")],
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
        if mode != core_pb2.JointCommand.POSITION:
            raise NotImplementedError(
                "Habitat-Sim Panda support currently accepts POSITION commands only"
            )
        if len(names) != len(data):
            raise ValueError("Joint command names and data must have the same length")

        positions = list(self._robot.joint_positions)
        joints_by_name = {info.name: info for info in self._joint_infos}
        for name, value in zip(names, data, strict=True):
            info = joints_by_name.get(name)
            if info is None:
                raise ValueError(f"Unknown Habitat-Sim joint '{name}'")
            positions[info.position_index] = min(max(value, info.lower_limit), info.upper_limit)
        self._robot.joint_positions = positions

    def servo_control_stream(
        self,
        request_iterator: Iterator[core_pb2.ServoCommand],
    ) -> Iterator[common_pb2.JointState]:
        del request_iterator
        raise NotImplementedError("Robot servo control is not supported for Habitat-Sim")

    def get_end_effector_state(self, group: str) -> core_pb2.EndEffectorState:
        del group
        raise NotImplementedError("End-effector state is not supported for Habitat-Sim")

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
            sim_cfg.enable_physics = self._uses_panda_robot

        agent_cfg = habitat_sim.AgentConfiguration()
        if self._camera_enabled:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = self._camera.name
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [self._camera.height, self._camera.width]
            sensor_spec.position = list(self._camera.position)
            agent_cfg.sensor_specifications = [sensor_spec]

        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    @property
    def _camera_enabled(self) -> bool:
        if self._enable_camera_override is not None:
            return self._enable_camera_override
        return not self._uses_panda_robot

    @property
    def _uses_panda_robot(self) -> bool:
        return self._requested_robot_name == "panda"

    def _default_camera(self) -> CameraConfig:
        if self._uses_panda_robot:
            return CameraConfig(position=PANDA_CAMERA_POSITION)
        return CameraConfig()

    def _load_robot(self) -> Any | None:
        if not self._uses_panda_robot:
            return None
        if self._sim is None:
            return None
        robot = self._sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            str(self._prepare_panda_urdf()),
            fixed_base=True,
            maintain_link_order=True,
        )
        self._joint_infos = self._build_robot_joint_infos(robot)
        self._apply_panda_ready_state(robot)
        return robot

    def _prepare_panda_urdf(self) -> Path:
        source = Path(
            "drivers_sim/gazebo-11/assets/robots/franka_panda/"
            "panda_description/urdf/panda.urdf"
        ).resolve()
        mesh_root = source.parents[1]
        text = source.read_text()
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
        return str(path.resolve()) if path.exists() else scene_path

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
