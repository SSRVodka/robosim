"""Tests for SimulationService backend dispatch."""

from __future__ import annotations

from control_stubs.common_pb2 import Empty, Point, Pose, Quaternion
from control_stubs.simulation_pb2 import ObjectState
from robosim.grpc_server.simulation import SimulationServicer


class DummyContext:
    def __init__(self) -> None:
        self.code = None

    def set_code(self, code) -> None:
        self.code = code


class SimulationControlBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def step_physics(self) -> None:
        self.calls.append("step")

    def pause(self) -> None:
        self.calls.append("pause")

    def resume(self) -> None:
        self.calls.append("resume")

    def set_object_pose(self, object_name: str, pose: Pose) -> None:
        self.calls.append(f"pose:{object_name}:{pose.position.x}")


def test_simulation_service_dispatches_optional_backend_controls() -> None:
    backend = SimulationControlBackend()
    servicer = SimulationServicer(backend)  # type: ignore[arg-type]
    context = DummyContext()

    servicer.StepPhysics(Empty(), context)
    servicer.Pause(Empty(), context)
    servicer.Resume(Empty(), context)
    servicer.SetObjectPose(
        ObjectState(
            object_name="mug",
            pose=Pose(
                position=Point(x=0.1, y=0.2, z=0.3),
                orientation=Quaternion(w=1.0),
            ),
        ),
        context,
    )

    assert context.code is None
    assert backend.calls == ["step", "pause", "resume", "pose:mug:0.1"]
