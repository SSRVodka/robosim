"""Client-shaped adapter for running navigation against an existing backend."""

from __future__ import annotations

from control_stubs import common_pb2, robot_core_pb2, sensing_pb2
from robosim.core.backend import SimulatorBackend


class BackendRobosimClient:
    """Tiny adapter with the RobosimClient subset used by navigation code."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self.mobility = _MobilityAdapter(backend)
        self.sensing = _SensingAdapter(backend)
        self.robot_core = _RobotCoreAdapter(backend)


class _MobilityAdapter:
    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        return self._backend.get_robot_pose_in_map()


class _SensingAdapter:
    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def get_sensors(self, sensor_names: list[str]) -> sensing_pb2.SensorData:
        return self._backend.get_sensors(sensor_names)


class _RobotCoreAdapter:
    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: robot_core_pb2.JointCommand.ControlMode,
        jmg_name: str | None = None,
    ) -> common_pb2.Status:
        self._backend.set_joint_target(names, data, mode, jmg_name)
        return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)

    def emergency_stop(self) -> common_pb2.Status:
        self._backend.emergency_stop()
        return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
