"""Tests for MuJoCo backend."""

from pathlib import Path

import pytest

from robosim.backends.mujoco.backend import MuJoCoBackend

MUJOCO_ASSETS_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim"
    / "mujoco"
    / "assets"
    / "robots"
    / "franka_panda"
)
SCENE_PATH = MUJOCO_ASSETS_PATH / "scene.xml"


class TestMuJoCoBackend:
    """Test MuJoCoBackend class."""

    @pytest.fixture
    def backend(self) -> MuJoCoBackend:
        instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
        try:
            yield instance
        finally:
            instance.shutdown()

    def test_backend_initialization(self, backend: MuJoCoBackend) -> None:
        """Test backend initializes correctly."""
        assert backend.robot_name == "panda"
        assert backend._model is not None
        assert backend._data is not None

    def test_capabilities(self, backend: MuJoCoBackend) -> None:
        """Test capabilities are detected."""
        caps = backend.capabilities
        assert caps is not None
        # Should have emergency stop capability
        from robosim.core.capabilities import Capability
        assert caps & Capability.EMERGENCY_STOP
        # Should have joint read capability
        assert caps & Capability.JOINT_READ

    def test_list_sensors(self, backend: MuJoCoBackend) -> None:
        """Test listing sensors."""
        sensor_list = backend.list_sensors()
        assert sensor_list is not None
        assert len(sensor_list.entries) >= 0

    def test_get_sensors_empty_names(self, backend: MuJoCoBackend) -> None:
        """Test getting sensors with empty names list."""
        sensor_data = backend.get_sensors([])
        assert sensor_data is not None

    def test_get_robot_state(self, backend: MuJoCoBackend) -> None:
        """Test getting robot state."""
        state = backend.get_robot_state()
        assert state is not None
        assert len(state.name) > 0

    def test_get_robot_spec(self, backend: MuJoCoBackend) -> None:
        """Test getting robot specification."""
        spec = backend.get_robot_spec()
        assert spec is not None
        assert spec.robot_name == "panda"
        assert len(spec.joints) > 0

    def test_set_joint_target_position(self, backend: MuJoCoBackend) -> None:
        """Test setting joint target in position mode."""
        from control_stubs.robot_core_pb2 import JointCommand

        # Get current state first
        initial_state = backend.get_robot_state()
        joint_name = initial_state.name[0]
        initial_pos = initial_state.position[0]

        # Set a new position
        new_pos = initial_pos + 0.1
        backend.set_joint_target(
            [joint_name],
            [new_pos],
            JointCommand.ControlMode.POSITION,
            group="panda_arm",
        )

        command_state = backend.get_joint_command_state()
        command_positions = dict(
            zip(command_state.name, command_state.position, strict=True)
        )
        assert command_positions[joint_name] == pytest.approx(new_pos)

    def test_reset_world(self, backend: MuJoCoBackend) -> None:
        """Test resetting world."""
        backend.reset_world(seed=42, randomization_params={})
        assert backend._data is not None

    def test_emergency_stop(self, backend: MuJoCoBackend) -> None:
        """Test emergency stop."""
        backend.emergency_stop()
        # Should not raise any exception

    def test_navigation_not_supported(self, backend: MuJoCoBackend) -> None:
        """Test that navigation raises NotImplementedError."""
        from control_stubs import mobility_ai_pb2 as mobility_pb2

        with pytest.raises(NotImplementedError):
            backend.get_robot_pose_in_map()

        with pytest.raises(NotImplementedError):
            backend.navigate_to(mobility_pb2.NavGoal())

    def test_shutdown(self, backend: MuJoCoBackend) -> None:
        """Test shutdown does not raise."""
        backend.shutdown()

    def test_joint_info_populated(self, backend: MuJoCoBackend) -> None:
        """Test that joint info is properly populated."""
        assert len(backend._joint_infos) > 0
        for info in backend._joint_infos:
            assert info.name
            assert info.qpos_adr >= 0
            assert info.qvel_adr >= 0


class TestMuJoCoBackendWithDefaultScene:
    """Test MuJoCoBackend with default scene path."""

    def test_default_scene_path(self) -> None:
        """Test using default scene path."""
        backend = MuJoCoBackend(str(SCENE_PATH), headless=True)
        try:
            assert backend.robot_name == "panda"
        finally:
            backend.shutdown()
