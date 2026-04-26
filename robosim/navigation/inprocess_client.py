"""Client-shaped adapter that runs a MuJoCo backend in the current process."""

from __future__ import annotations

from control_stubs import common_pb2, robot_core_pb2, sensing_pb2
from robosim.backends.mujoco.backend import MuJoCoBackend


class InProcessRobosimClient:
    """Tiny adapter with the subset of RobosimClient used by navigation demos."""

    def __init__(self, scene_path: str, headless: bool = True) -> None:
        self._backend = MuJoCoBackend(scene_path, headless=headless)
        self.mobility = _MobilityAdapter(self._backend)
        self.sensing = _SensingAdapter(self._backend)
        self.robot_core = _RobotCoreAdapter(self._backend)

    def close(self) -> None:
        self._backend.shutdown()


class _MobilityAdapter:
    def __init__(self, backend: MuJoCoBackend) -> None:
        self._backend = backend

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        return self._backend.get_robot_pose_in_map()


class _SensingAdapter:
    def __init__(self, backend: MuJoCoBackend) -> None:
        self._backend = backend

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        return self._backend.list_sensors()

    def get_sensors(self, sensor_names: list[str]) -> sensing_pb2.SensorData:
        return self._backend.get_sensors(sensor_names)


class _RobotCoreAdapter:
    def __init__(self, backend: MuJoCoBackend) -> None:
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
