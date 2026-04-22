"""Tests for RoboSim core module."""

from robosim.core.capabilities import Capability


class TestCapability:
    """Test capability flags."""

    def test_capability_none(self) -> None:
        assert Capability.NONE.value == 0

    def test_capability_joins(self) -> None:
        combined = Capability.SENSOR_CAMERA | Capability.SENSOR_LIDAR
        expected = Capability.SENSOR_CAMERA | Capability.SENSOR_LIDAR
        assert combined == expected

    def test_capability_servo(self) -> None:
        expected = Capability.JOINT_READ | Capability.JOINT_WRITE | Capability.END_EFFECTOR_READ
        assert Capability.SERVO_CAPABLE == expected

    def test_capability_navigable(self) -> None:
        expected = Capability.NAVIGATION | Capability.JOINT_READ
        assert Capability.NAVIGABLE == expected

    def test_capability_sensor_all(self) -> None:
        expected = (
            Capability.SENSOR_CAMERA | Capability.SENSOR_LIDAR |
            Capability.SENSOR_IMU | Capability.SENSOR_JOINT |
            Capability.SENSOR_ODOMETRY
        )
        assert Capability.SENSOR_ALL == expected


class TestSimulatorBackend:
    """Test SimulatorBackend abstract class."""

    def test_backend_is_abc(self) -> None:
        from robosim.core.backend import SimulatorBackend

        assert hasattr(SimulatorBackend, "get_robot_state")
        assert hasattr(SimulatorBackend, "get_robot_spec")
        assert hasattr(SimulatorBackend, "set_joint_target")
        assert hasattr(SimulatorBackend, "list_sensors")
        assert hasattr(SimulatorBackend, "navigate_to")
        assert hasattr(SimulatorBackend, "emergency_stop")
