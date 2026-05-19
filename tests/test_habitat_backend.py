"""Tests for the Habitat-Sim backend wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.habitat.backend import HabitatSimBackend
from robosim.core.capabilities import Capability


class FakeSimulatorConfiguration:
    def __init__(self) -> None:
        self.scene_id = ""
        self.enable_physics = True
        self.create_renderer = True


class FakeCameraSensorSpec:
    def __init__(self) -> None:
        self.uuid = ""
        self.sensor_type = None
        self.resolution: list[int] = []
        self.position: list[float] = []


class FakeAgentConfiguration:
    def __init__(self) -> None:
        self.sensor_specifications: list[FakeCameraSensorSpec] = []


class FakeConfiguration:
    def __init__(
        self,
        sim_cfg: FakeSimulatorConfiguration,
        agent_cfgs: list[FakeAgentConfiguration],
    ) -> None:
        self.sim_cfg = sim_cfg
        self.agent_cfgs = agent_cfgs


class FakeSimulator:
    last_config: FakeConfiguration | None = None

    def __init__(self, config: FakeConfiguration) -> None:
        self.config = config
        self.closed = False
        self.reset_count = 0
        self.articulated_object_manager = FakeArticulatedObjectManager()
        self.agent = FakeAgent()
        FakeSimulator.last_config = config

    def get_sensor_observations(self):
        sensor_name = self.config.agent_cfgs[0].sensor_specifications[0].uuid
        return {
            sensor_name: np.full((4, 6, 4), [10, 20, 30, 255], dtype=np.uint8),
        }

    def reset(self) -> None:
        self.reset_count += 1

    def get_articulated_object_manager(self):
        return self.articulated_object_manager

    def get_agent(self, agent_id: int):
        assert agent_id == 0
        return self.agent

    def close(self) -> None:
        self.closed = True


class FakeAgentState:
    def __init__(self) -> None:
        self.position = [0.0, 0.0, 0.0]
        self.rotation = (0.0, 0.0, 0.0, 1.0)


class FakeAgent:
    def __init__(self) -> None:
        self.state = FakeAgentState()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeArticulatedObjectManager:
    def __init__(self) -> None:
        self.loaded_urdf: str | None = None
        self.robot = FakeArticulatedObject()

    def add_articulated_object_from_urdf(
        self,
        filepath: str,
        fixed_base: bool,
        maintain_link_order: bool,
    ):
        self.loaded_urdf = filepath
        self.robot.fixed_base = fixed_base
        self.robot.maintain_link_order = maintain_link_order
        return self.robot


class FakeJointType:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeArticulatedObject:
    def __init__(self) -> None:
        self.fixed_base = False
        self.maintain_link_order = False
        self.joint_positions = [0.0 for _ in range(9)]
        self.joint_position_limits = (
            [-3.0, -2.0, -3.0, -3.2, -3.0, -0.1, -3.0, 0.0, 0.0],
            [3.0, 2.0, 3.0, 0.1, 3.0, 3.9, 3.0, 0.04, 0.04],
        )
        self._joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

    def get_link_ids(self):
        return list(range(len(self._joint_names)))

    def get_link_num_joint_pos(self, link_id: int) -> int:
        del link_id
        return 1

    def get_link_joint_pos_offset(self, link_id: int) -> int:
        return link_id

    def get_link_joint_name(self, link_id: int) -> str:
        return self._joint_names[link_id]

    def get_link_joint_type(self, link_id: int):
        return FakeJointType("Prismatic" if link_id >= 7 else "Revolute")

    def get_link_scene_node(self, link_id: int):
        return FakeSceneNode(self, link_id)


class FakeVector3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeQuaternion:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class FakeTransform:
    def __init__(self, robot: FakeArticulatedObject, link_id: int) -> None:
        self._robot = robot
        self._link_id = link_id

    def translation(self):
        del self._link_id
        return FakeVector3(
            self._robot.joint_positions[0],
            self._robot.joint_positions[1],
            self._robot.joint_positions[2],
        )

    @property
    def rotation(self):
        return FakeQuaternion()


class FakeSceneNode:
    def __init__(self, robot: FakeArticulatedObject, link_id: int) -> None:
        self._robot = robot
        self._link_id = link_id

    def absolute_transformation(self):
        return FakeTransform(self._robot, self._link_id)


class FakeViewerProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> None:
        del timeout

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def fake_habitat_sim(monkeypatch: pytest.MonkeyPatch):
    fake_module = SimpleNamespace(
        SimulatorConfiguration=FakeSimulatorConfiguration,
        CameraSensorSpec=FakeCameraSensorSpec,
        AgentConfiguration=FakeAgentConfiguration,
        Configuration=FakeConfiguration,
        Simulator=FakeSimulator,
        AgentState=FakeAgentState,
        SensorType=SimpleNamespace(COLOR="color"),
    )
    monkeypatch.setitem(sys.modules, "habitat_sim", fake_module)
    return fake_module


def test_habitat_backend_lists_and_renders_camera(fake_habitat_sim) -> None:
    backend = HabitatSimBackend(scene_path="/tmp/example.glb")

    try:
        assert backend.capabilities == Capability.SENSOR_CAMERA | Capability.SIMULATION_CONTROL
        assert backend.robot_name == "habitat_camera"
        assert backend.get_robot_spec().robot_name == "habitat_camera"
        assert backend.get_robot_state().name == []

        sensors = backend.list_sensors()
        assert len(sensors.entries) == 1
        assert sensors.entries[0].name == "habitat_rgb"

        data = backend.get_sensors(["habitat_rgb"])
        assert len(data.images) == 1
        image = data.images[0]
        assert image.width == 6
        assert image.height == 4
        assert image.encoding == "rgb8"
        assert image.step == 18
        assert image.data == np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8).tobytes()

        config = FakeSimulator.last_config
        assert config is not None
        assert config.sim_cfg.scene_id == "/tmp/example.glb"
        assert config.sim_cfg.enable_physics is False
    finally:
        backend.shutdown()


def test_habitat_backend_reset_delegates_to_simulator(fake_habitat_sim) -> None:
    backend = HabitatSimBackend()

    try:
        backend.reset_world(seed=123, randomization_params={"ignored": 1.0})
        assert backend._sim is not None
        assert backend._sim.reset_count == 1
    finally:
        backend.shutdown()


def test_habitat_backend_loads_panda_and_sets_joint_targets(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(HabitatSimBackend, "_prepare_panda_urdf", lambda self: urdf)

    backend = HabitatSimBackend(robot_name="panda")

    try:
        assert backend.capabilities & Capability.JOINT_READ
        assert backend.capabilities & Capability.JOINT_WRITE
        assert backend.robot_name == "panda"

        spec = backend.get_robot_spec()
        assert spec.robot_name == "panda"
        assert [group.name for group in spec.joint_model_groups] == [
            "panda_arm",
            "panda_hand",
            "panda_arm_hand",
        ]
        assert spec.joint_model_groups[0].joint_names == [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        state = backend.get_robot_state()
        assert state.name[:2] == ["panda_joint1", "panda_joint2"]
        assert state.position[1] == pytest.approx(-0.785)

        backend.set_joint_target(
            names=["panda_joint2", "panda_finger_joint1"],
            data=[0.5, 0.5],
            mode=core_pb2.JointCommand.POSITION,
        )
        state = backend.get_robot_state()
        position_by_name = dict(zip(state.name, state.position, strict=True))
        assert position_by_name["panda_joint2"] == pytest.approx(0.5)
        assert position_by_name["panda_finger_joint1"] == pytest.approx(0.04)
    finally:
        backend.shutdown()


def test_habitat_backend_rejects_unknown_robot_name(fake_habitat_sim) -> None:
    del fake_habitat_sim

    with pytest.raises(ValueError, match="supports only robot_name='panda'"):
        HabitatSimBackend(robot_name="xxx")


def test_habitat_backend_can_render_panda_when_camera_enabled(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(HabitatSimBackend, "_prepare_panda_urdf", lambda self: urdf)

    backend = HabitatSimBackend(robot_name="panda", enable_camera=True)

    try:
        assert backend.capabilities & Capability.SENSOR_CAMERA
        sensors = backend.list_sensors()
        assert sensors.entries[0].name == "habitat_rgb"

        data = backend.get_sensors(["habitat_rgb"])
        assert len(data.images) == 1
        assert data.images[0].width == 6

        config = FakeSimulator.last_config
        assert config is not None
        assert config.sim_cfg.create_renderer is True
        sensor_spec = config.agent_cfgs[0].sensor_specifications[0]
        assert sensor_spec.position == [0.0, 0.0, 0.0]
    finally:
        backend.shutdown()


def test_habitat_backend_can_move_camera_with_set_object_pose(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    del fake_habitat_sim
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(HabitatSimBackend, "_prepare_panda_urdf", lambda self: urdf)

    backend = HabitatSimBackend(robot_name="panda", enable_camera=True)

    try:
        pose = common_pb2.Pose(
            position=common_pb2.Point(x=1.0, y=2.0, z=3.0),
            orientation=common_pb2.Quaternion(w=1.0),
        )
        backend.set_object_pose("habitat_rgb", pose)

        assert backend._sim.agent.state.position == [1.0, 2.0, 3.0]
    finally:
        backend.shutdown()


def test_habitat_backend_exposes_panda_end_effector(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    del fake_habitat_sim
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(HabitatSimBackend, "_prepare_panda_urdf", lambda self: urdf)

    backend = HabitatSimBackend(robot_name="panda")

    try:
        spec = backend.get_robot_spec()
        ee = spec.joint_model_groups[0].end_effectors[0]
        assert ee.name == "hand"
        assert ee.parent_jmg_name == "panda_arm"

        state = backend.get_end_effector_state("panda_arm")
        assert state.pose_stamped.header.frame_id == "world"
        assert state.pose_stamped.pose.orientation.w == 1.0
    finally:
        backend.shutdown()


def test_habitat_backend_servo_control_stream_accepts_twist(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    del fake_habitat_sim
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(HabitatSimBackend, "_prepare_panda_urdf", lambda self: urdf)

    backend = HabitatSimBackend(robot_name="panda")

    try:
        before = backend.get_robot_state()
        command = core_pb2.ServoCommand(
            twist_cmd=core_pb2.TwistCommand(
                twist=common_pb2.TwistStamped(
                    twist=common_pb2.Twist(
                        linear=common_pb2.Point(x=0.1),
                    )
                ),
                target_ee=core_pb2.EESpec(name="hand", parent_jmg_name="panda_arm"),
            )
        )

        states = list(backend.servo_control_stream(iter([command])))

        assert len(states) == 1
        after = states[0]
        assert after.position[0] > before.position[0]
    finally:
        backend.shutdown()


def test_habitat_backend_display_mode_launches_viewer(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_habitat_sim
    launched: dict[str, Any] = {}
    process = FakeViewerProcess()

    monkeypatch.setattr("shutil.which", lambda name: "/tmp/viewer" if name == "viewer" else None)

    def fake_popen(args, env):
        launched["args"] = args
        launched["env"] = env
        return process

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    backend = HabitatSimBackend(scene_path="/tmp/example.glb", headless=False)

    try:
        assert backend._sim is None
        assert launched["args"] == ["/tmp/viewer", "/tmp/example.glb"]
        assert launched["env"]["LIBGL_ALWAYS_SOFTWARE"] == "1"
        assert launched["env"]["MESA_GL_VERSION_OVERRIDE"] == "4.1"
        with pytest.raises(NotImplementedError, match="display viewer mode"):
            backend.get_sensors([])
    finally:
        backend.shutdown()

    assert process.terminated


def test_habitat_backend_display_mode_rejects_panda(fake_habitat_sim) -> None:
    del fake_habitat_sim

    with pytest.raises(NotImplementedError, match="Simulator API"):
        HabitatSimBackend(scene_path="/tmp/example.glb", headless=False, robot_name="panda")


def test_habitat_backend_display_mode_requires_scene(fake_habitat_sim) -> None:
    del fake_habitat_sim

    with pytest.raises(ValueError, match="requires a mesh scene"):
        HabitatSimBackend(headless=False)


def test_habitat_backend_display_mode_rejects_mjcf_scene(
    fake_habitat_sim,
    tmp_path,
) -> None:
    del fake_habitat_sim
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco model='bad_for_habitat'></mujoco>")

    with pytest.raises(ValueError, match="cannot load MuJoCo MJCF"):
        HabitatSimBackend(scene_path=str(scene), headless=False)


def test_habitat_backend_resolves_external_drivers_sim_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    drivers_root = tmp_path / "drivers_sim"
    urdf_dir = (
        drivers_root
        / "gazebo-11/assets/robots/franka_panda/panda_description/urdf"
    )
    urdf_dir.mkdir(parents=True)
    urdf = urdf_dir / "panda.urdf"
    urdf.write_text(
        "<robot name='panda'>"
        "<link name='panda_link0'>"
        "<visual><geometry>"
        "<mesh filename='package://franka_panda_desc/meshes/visual/link0.dae'/>"
        "</geometry></visual>"
        "</link>"
        "</robot>"
    )
    monkeypatch.setenv("ROBOSIM_DRIVERS_SIM_ROOT", str(drivers_root))
    backend = HabitatSimBackend.__new__(HabitatSimBackend)

    prepared = backend._prepare_panda_urdf()

    try:
        assert str(drivers_root / "gazebo-11/assets/robots/franka_panda/panda_description") in (
            prepared.read_text()
        )
    finally:
        prepared.unlink(missing_ok=True)


def test_habitat_backend_resolves_scene_relative_to_external_drivers_root(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    drivers_root = tmp_path / "drivers_sim"
    scene = drivers_root / "habitat/assets/worlds/apartment.glb"
    scene.parent.mkdir(parents=True)
    scene.write_text("fake glb")
    monkeypatch.setenv("ROBOSIM_DRIVERS_SIM_ROOT", str(drivers_root))

    backend = HabitatSimBackend(scene_path="habitat/assets/worlds/apartment.glb")

    try:
        config = FakeSimulator.last_config
        assert config is not None
        assert config.sim_cfg.scene_id == str(scene.resolve())
    finally:
        backend.shutdown()


def test_habitat_backend_resolves_scene_with_drivers_sim_prefix(
    fake_habitat_sim,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    drivers_root = tmp_path / "drivers_sim"
    scene = drivers_root / "habitat/assets/worlds/apartment.glb"
    scene.parent.mkdir(parents=True)
    scene.write_text("fake glb")
    monkeypatch.setenv("ROBOSIM_DRIVERS_SIM_ROOT", str(drivers_root))

    backend = HabitatSimBackend(scene_path="drivers_sim/habitat/assets/worlds/apartment.glb")

    try:
        config = FakeSimulator.last_config
        assert config is not None
        assert config.sim_cfg.scene_id == str(scene.resolve())
    finally:
        backend.shutdown()
