"""Tests for thread-bound backend execution."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from control_stubs import common_pb2, mobility_ai_pb2, sensing_pb2
from robosim.core.backend import SimulatorBackend
from robosim.core.capabilities import Capability
from robosim.server import ThreadBoundBackendProxy


class RecordingBackend(SimulatorBackend):
    def __init__(self) -> None:
        self.thread_ids = [threading.get_ident()]

    def _record(self) -> None:
        self.thread_ids.append(threading.get_ident())

    @property
    def capabilities(self) -> Capability:
        self._record()
        return Capability.SIMULATION_CONTROL

    @property
    def robot_name(self) -> str:
        self._record()
        return "recording"

    @property
    def headless_mode(self) -> bool:
        self._record()
        return True

    def set_headless_mode(self, enabled: bool) -> None:
        del enabled
        self._record()

    def get_robot_state(self) -> common_pb2.JointState:
        self._record()
        return common_pb2.JointState(name=["joint"])

    def get_robot_spec(self) -> Any:
        self._record()
        return object()

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: Any,
        group: str | None = None,
    ) -> None:
        del names, data, mode, group
        self._record()

    def servo_control_stream(
        self,
        request_iterator: Iterator[Any],
    ) -> Iterator[common_pb2.JointState]:
        del request_iterator
        self._record()
        yield common_pb2.JointState(name=["servo"])

    def get_end_effector_state(self, group: str) -> Any:
        del group
        self._record()
        return object()

    def get_joint_command_state(self) -> common_pb2.JointState:
        self._record()
        return common_pb2.JointState(name=["command"])

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        self._record()
        return sensing_pb2.SensorMetaList()

    def get_sensors(self, names: list[str]) -> sensing_pb2.SensorData:
        del names
        self._record()
        return sensing_pb2.SensorData()

    def stream_sensors(self, names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        del names
        self._record()
        yield sensing_pb2.SensorData()

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        self._record()
        return common_pb2.PoseStamped()

    def navigate_to(self, goal: mobility_ai_pb2.NavGoal) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        del goal
        self._record()
        yield mobility_ai_pb2.TaskFeedback()

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        del seed, randomization_params
        self._record()

    def emergency_stop(self) -> None:
        self._record()

    def shutdown(self) -> None:
        self._record()


def test_thread_bound_backend_proxy_runs_backend_on_one_thread() -> None:
    created: dict[str, RecordingBackend] = {}

    def make_backend() -> RecordingBackend:
        backend = RecordingBackend()
        created["backend"] = backend
        return backend

    proxy = ThreadBoundBackendProxy(make_backend)

    try:
        assert proxy.robot_name == "recording"
        assert proxy.capabilities == Capability.SIMULATION_CONTROL
        assert proxy.get_robot_state().name == ["joint"]
        assert len(list(proxy.stream_sensors(["camera"]))) == 1
    finally:
        proxy.shutdown()

    backend = created["backend"]
    assert len(set(backend.thread_ids)) == 1
