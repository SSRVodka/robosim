#!/usr/bin/env python3
"""gRPC server main entry point."""

from __future__ import annotations

import argparse
import asyncio
import signal
from concurrent import futures
from pathlib import Path
from typing import Any, Callable, TypeVar

from grpc import aio as grpc_aio

from control_stubs import (
    common_pb2,
)
from control_stubs import (
    mobility_ai_pb2_grpc as mobility_grpc,
)
from control_stubs import (
    policy_pb2_grpc as policy_grpc,
)
from control_stubs import (
    robot_core_pb2_grpc as core_grpc,
)
from control_stubs import (
    robot_data_pb2_grpc as data_grpc,
)
from control_stubs import (
    sensing_pb2_grpc as sensing_grpc,
)
from control_stubs import (
    simulation_pb2_grpc as sim_grpc,
)
from robosim.core.activity import ActivityCoordinator
from robosim.core.backend import SimulatorBackend
from robosim.core.recorder import DataRecorder
from robosim.grpc_server import (
    MobilityServicer,
    PolicyInferenceServicer,
    RobotCoreServicer,
    RobotDataServicer,
    SensingServicer,
    SimulationServicer,
)

DATA_REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_UNAVAILABLE_MESSAGE = (
    "LeRobot recorder/policy is unavailable in this environment. "
    "Install a compatible lerobot version to use this service."
)
T = TypeVar("T")


class UnavailableDataRecorder(DataRecorder):
    """Data recorder stub used when LeRobot cannot be imported."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def episode_start(self, options: Any) -> Any:
        del options
        raise NotImplementedError(self._reason)

    def episode_end(self) -> common_pb2.Status:
        raise NotImplementedError(self._reason)

    def episode_replay(self, info: Any) -> common_pb2.Status:
        del info
        raise NotImplementedError(self._reason)


class UnavailablePolicyRunner:
    """Policy runner stub used when LeRobot cannot be imported."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def load_policy(self, request: Any) -> common_pb2.Status:
        del request
        raise NotImplementedError(self._reason)

    def start_policy(self, request: Any) -> common_pb2.Status:
        del request
        raise NotImplementedError(self._reason)

    def stop_policy(self) -> common_pb2.Status:
        return common_pb2.Status(
            code=common_pb2.STATUS_SUCCESS,
            message="no policy is running",
        )

    def get_status(self) -> Any:
        from control_stubs import policy_pb2

        return policy_pb2.PolicyStatus(
            status=common_pb2.Status(
                code=common_pb2.STATUS_FAILURE,
                message=self._reason,
            ),
            loaded=False,
            running=False,
        )

    def notify_world_reset(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class ThreadBoundBackendProxy(SimulatorBackend):
    """Run a backend and all of its calls on one dedicated thread.

    Some simulator APIs bind rendering contexts to the creating thread. The
    proxy keeps those backends usable from gRPC's worker threads by serializing
    every backend call through the same single-thread executor that constructs
    the backend.
    """

    def __init__(self, factory: Callable[[], SimulatorBackend]) -> None:
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._backend = self._executor.submit(factory).result()
        self._closed = False

    def _call(self, fn: Callable[[SimulatorBackend], T]) -> T:
        if self._closed:
            raise RuntimeError("Backend has been shut down")
        return self._executor.submit(lambda: fn(self._backend)).result()

    @property
    def capabilities(self) -> Any:
        return self._call(lambda backend: backend.capabilities)

    @property
    def robot_name(self) -> str:
        return self._call(lambda backend: backend.robot_name)

    @property
    def headless_mode(self) -> bool:
        return self._call(lambda backend: backend.headless_mode)

    def set_headless_mode(self, enabled: bool) -> None:
        return self._call(lambda backend: backend.set_headless_mode(enabled))

    def get_robot_state(self) -> Any:
        return self._call(lambda backend: backend.get_robot_state())

    def get_robot_spec(self) -> Any:
        return self._call(lambda backend: backend.get_robot_spec())

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: Any,
        group: str | None = None,
    ) -> None:
        return self._call(lambda backend: backend.set_joint_target(names, data, mode, group))

    def servo_control_stream(self, request_iterator: Any) -> Any:
        iterator = self._call(lambda backend: backend.servo_control_stream(request_iterator))
        while True:
            has_item, item = self._call(lambda _backend: self._next_iterator(iterator))
            if not has_item:
                return
            yield item

    def get_end_effector_state(self, group: str) -> Any:
        return self._call(lambda backend: backend.get_end_effector_state(group))

    def get_joint_command_state(self) -> Any:
        return self._call(lambda backend: backend.get_joint_command_state())

    def list_sensors(self) -> Any:
        return self._call(lambda backend: backend.list_sensors())

    def get_sensors(self, names: list[str]) -> Any:
        return self._call(lambda backend: backend.get_sensors(names))

    def stream_sensors(self, names: list[str]) -> Any:
        iterator = self._call(lambda backend: backend.stream_sensors(names))
        while True:
            has_item, item = self._call(lambda _backend: self._next_iterator(iterator))
            if not has_item:
                return
            yield item

    def get_robot_pose_in_map(self) -> Any:
        return self._call(lambda backend: backend.get_robot_pose_in_map())

    def navigate_to(self, goal: Any) -> Any:
        iterator = self._call(lambda backend: backend.navigate_to(goal))
        while True:
            has_item, item = self._call(lambda _backend: self._next_iterator(iterator))
            if not has_item:
                return
            yield item

    def reset_world(self, seed: int, randomization_params: dict[str, float]) -> None:
        return self._call(lambda backend: backend.reset_world(seed, randomization_params))

    def set_object_pose(self, object_name: str, pose: Any) -> None:
        return self._call(lambda backend: getattr(backend, "set_object_pose")(object_name, pose))

    def emergency_stop(self) -> None:
        return self._call(lambda backend: backend.emergency_stop())

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._backend.shutdown).result()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)

    @staticmethod
    def _next_iterator(iterator: Any) -> tuple[bool, Any]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None


def create_lerobot_services(
    backend: SimulatorBackend,
    activity: ActivityCoordinator,
) -> tuple[DataRecorder, Any]:
    """Create LeRobot integrations, or stubs if the installed version is incompatible."""
    try:
        from robosim.core.impl.policy_lerobot import LerobotPolicyRunner
        from robosim.core.impl.recorder_lerobot import LerobotDataRecorder
    except ImportError as exc:
        reason = f"{LEROBOT_UNAVAILABLE_MESSAGE} Import error: {exc}"
        return UnavailableDataRecorder(reason), UnavailablePolicyRunner(reason)

    return (
        LerobotDataRecorder(DATA_REPO_ROOT, backend, activity_coordinator=activity),
        LerobotPolicyRunner(DATA_REPO_ROOT, backend, activity_coordinator=activity),
    )


def create_server(
    backend: SimulatorBackend,
    recorder: DataRecorder,
    policy_runner: Any,
    port: int = 50051,
    max_workers: int = 10,
) -> grpc_aio.Server:
    """Create and configure the gRPC server."""
    server = grpc_aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    sim_grpc.add_SimulationServiceServicer_to_server(
        SimulationServicer(backend, policy_runner=policy_runner), server
    )
    sensing_grpc.add_SensingServiceServicer_to_server(SensingServicer(backend), server)
    core_grpc.add_RobotCoreServiceServicer_to_server(RobotCoreServicer(backend), server)
    data_grpc.add_RobotDataServiceServicer_to_server(RobotDataServicer(recorder), server)
    policy_grpc.add_PolicyInferenceServiceServicer_to_server(
        PolicyInferenceServicer(policy_runner), server
    )
    mobility_grpc.add_MobilityServiceServicer_to_server(MobilityServicer(backend), server)

    server.add_insecure_port(f"[::]:{port}")
    return server


async def serve_async(
    backend_type: str,
    robot_name: str = "robot",
    port: int = 50051,
    scene: str | None = None,
    headless: bool = True,
    habitat_enable_camera: bool | None = None,
) -> None:
    """Run the gRPC server asynchronously."""
    backend: SimulatorBackend | None = None
    recorder: DataRecorder | None = None
    policy_runner: Any | None = None
    server: grpc_aio.Server | None = None

    async def shutdown_handler_async() -> None:
        """Async shutdown handler for gRPC server."""
        nonlocal server, backend, policy_runner
        print("\nReceived shutdown signal, stopping server...")
        if server is not None:
            await server.stop(grace=1.0)
        if policy_runner is not None:
            policy_runner.shutdown()
        if backend is not None:
            backend.shutdown()
        try:
            if backend_type == "gazebo":
                import rclpy

                rclpy.shutdown()
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    should_shutdown = False

    def shutdown_handler(sig: int, frame) -> None:
        nonlocal should_shutdown
        print(f"\nReceived signal {sig}, initiating shutdown...")
        should_shutdown = True
        loop.call_soon_threadsafe(lambda: asyncio.create_task(shutdown_handler_async()))

    try:
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        activity = ActivityCoordinator()
        if backend_type == "gazebo":
            import rclpy

            from robosim.backends import GazeboBackend

            rclpy.init()
            backend = GazeboBackend(robot_name=robot_name)
        elif backend_type == "mujoco":
            from robosim.backends import MuJoCoBackend

            backend = MuJoCoBackend(
                scene_path=scene or "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml",
                headless=headless,
            )
        elif backend_type == "habitat":
            from robosim.backends import HabitatSimBackend

            backend = ThreadBoundBackendProxy(
                lambda: HabitatSimBackend(
                    scene_path=scene,
                    headless=headless,
                    robot_name=robot_name if robot_name != "robot" else None,
                    enable_camera=habitat_enable_camera,
                )
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        recorder, policy_runner = create_lerobot_services(backend, activity)
        server = create_server(backend, recorder, policy_runner, port)

        await server.start()
        print(f"gRPC server started on port {port}")
        print(f"Backend: {backend_type}")
        print(f"Robot: {backend.robot_name}")
        print(f"Capabilities: {backend.capabilities}")
        print("Press Ctrl+C to stop")

        while not should_shutdown:
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Server error: {e}")
        if policy_runner is not None:
            policy_runner.shutdown()
        if backend is not None:
            backend.shutdown()
        try:
            if backend_type == "gazebo":
                import rclpy

                rclpy.shutdown()
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboSim gRPC Server")
    parser.add_argument(
        "--backend",
        type=str,
        default="gazebo",
        choices=["gazebo", "mujoco", "habitat"],
        help="Simulator backend type",
    )
    parser.add_argument(
        "--robot-name",
        "--robot",
        type=str,
        default="robot",
        dest="robot_name",
        help="Robot name for simulation",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC server port",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to the simulator scene file (for mujoco or habitat backends)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run simulator without viewer where supported",
    )
    parser.add_argument(
        "--habitat-enable-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable Habitat camera rendering explicitly. Useful for rendering Panda "
            "on a GPU/EGL machine; Panda keeps it disabled by default for CPU-only runs."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        serve_async(
            backend_type=args.backend,
            robot_name=args.robot_name,
            port=args.port,
            scene=args.scene,
            headless=args.headless,
            habitat_enable_camera=args.habitat_enable_camera,
        )
    )


if __name__ == "__main__":
    main()
