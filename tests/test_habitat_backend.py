"""Tests for the Habitat-Sim backend wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robosim.backends.habitat.backend import HabitatSimBackend
from robosim.core.capabilities import Capability


class FakeSimulatorConfiguration:
    def __init__(self) -> None:
        self.scene_id = ""
        self.enable_physics = True


class FakeCameraSensorSpec:
    def __init__(self) -> None:
        self.uuid = ""
        self.sensor_type = None
        self.resolution = []
        self.position = []


class FakeAgentConfiguration:
    def __init__(self) -> None:
        self.sensor_specifications = []


class FakeConfiguration:
    def __init__(self, sim_cfg, agent_cfgs) -> None:
        self.sim_cfg = sim_cfg
        self.agent_cfgs = agent_cfgs


class FakeSimulator:
    last_config = None

    def __init__(self, config) -> None:
        self.config = config
        self.closed = False
        self.reset_count = 0
        FakeSimulator.last_config = config

    def get_sensor_observations(self):
        sensor_name = self.config.agent_cfgs[0].sensor_specifications[0].uuid
        return {
            sensor_name: np.full((4, 6, 4), [10, 20, 30, 255], dtype=np.uint8),
        }

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


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
        assert config.sim_cfg.scene_id == "/tmp/example.glb"
        assert config.sim_cfg.enable_physics is False
    finally:
        backend.shutdown()


def test_habitat_backend_reset_delegates_to_simulator(fake_habitat_sim) -> None:
    backend = HabitatSimBackend()

    try:
        backend.reset_world(seed=123, randomization_params={"ignored": 1.0})
        assert backend._sim.reset_count == 1
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


def test_habitat_backend_display_mode_rejects_mjcf_scene(
    fake_habitat_sim,
    tmp_path,
) -> None:
    del fake_habitat_sim
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco model='bad_for_habitat'></mujoco>")

    with pytest.raises(ValueError, match="cannot load MuJoCo MJCF"):
        HabitatSimBackend(scene_path=str(scene), headless=False)
