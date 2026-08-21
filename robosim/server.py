#!/usr/bin/env python3
"""gRPC server main entry point."""

from __future__ import annotations

import argparse
import asyncio
import signal
from concurrent import futures
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence, cast

from grpc import aio as grpc_aio

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
from robosim.core.impl.policy_lerobot import LerobotPolicyRunner
from robosim.core.impl.recorder_lerobot import LerobotDataRecorder
from robosim.core.ros_environment import flush_ros_environment
from robosim.grpc_server import (
    MobilityServicer,
    PolicyInferenceServicer,
    RobotCoreServicer,
    RobotDataServicer,
    SensingServicer,
    SimulationServicer,
)

DATA_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_backend_class(backend_type: str) -> type[SimulatorBackend]:
    """Import only the backend selected for this server instance."""
    if backend_type == "gazebo":
        from robosim.backends.gazebo import GazeboBackend

        return GazeboBackend
    if backend_type == "mujoco":
        from robosim.backends.mujoco import MuJoCoBackend

        return MuJoCoBackend
    if backend_type == "pybullet":
        from robosim.backends.pybullet import PyBulletBackend

        return PyBulletBackend
    raise ValueError(f"Unknown backend type: {backend_type}")


def create_backend(
    *,
    backend_type: str,
    robot_name: str,
    scene: str | None,
    csd_manifest: str | None,
    headless: bool,
) -> SimulatorBackend:
    """Create a simulator backend for server startup."""
    backend_class = cast(Any, _load_backend_class(backend_type))
    if backend_type == "gazebo":
        if csd_manifest is not None:
            return backend_class.from_csd_realization_manifest_file(Path(csd_manifest))
        return backend_class(robot_name=robot_name)
    if backend_type == "mujoco":
        if csd_manifest is not None:
            return backend_class.from_csd_realization_manifest_file(
                Path(csd_manifest),
                headless=headless,
            )
        return backend_class(
            scene_path=scene or "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml",
            headless=headless,
        )
    if backend_type == "pybullet":
        if csd_manifest is not None:
            return backend_class.from_csd_realization_manifest_file(
                Path(csd_manifest),
                headless=headless,
            )
        return backend_class(scene_path=scene, headless=headless)
    raise AssertionError("backend type was validated before construction")


def create_server(
    backend: SimulatorBackend,
    recorder: LerobotDataRecorder,
    policy_runner: LerobotPolicyRunner,
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
    sensing_grpc.add_SensingServiceServicer_to_server(
        SensingServicer(backend), server
    )
    core_grpc.add_RobotCoreServiceServicer_to_server(
        RobotCoreServicer(backend), server
    )
    data_grpc.add_RobotDataServiceServicer_to_server(
        RobotDataServicer(recorder), server
    )
    policy_grpc.add_PolicyInferenceServiceServicer_to_server(
        PolicyInferenceServicer(policy_runner), server
    )
    mobility_grpc.add_MobilityServiceServicer_to_server(
        MobilityServicer(backend), server
    )

    server.add_insecure_port(f"[::]:{port}")
    return server


async def serve_async(
    backend_type: str,
    robot_name: str = "robot",
    port: int = 50051,
    scene: str | None = None,
    csd_manifest: str | None = None,
    headless: bool = True,
) -> None:
    """Run the gRPC server asynchronously."""
    backend: SimulatorBackend | None = None
    recorder: LerobotDataRecorder | None = None
    policy_runner: LerobotPolicyRunner | None = None
    server: grpc_aio.Server | None = None
    gazebo_runtime: Any | None = None
    rclpy: ModuleType | None = None

    async def shutdown() -> None:
        """Stop services and owned resources in dependency order."""
        nonlocal server, backend, policy_runner, gazebo_runtime
        if server is not None:
            await server.stop(grace=1.0)
        if policy_runner is not None:
            policy_runner.shutdown()
        if backend is not None:
            backend.shutdown()
        if gazebo_runtime is not None:
            gazebo_runtime.stop()
        if rclpy is not None:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    loop = asyncio.get_running_loop()
    shutdown_requested = asyncio.Event()

    def shutdown_handler(sig: int, frame) -> None:
        del frame
        print(f"\nReceived signal {sig}, initiating shutdown...")
        loop.call_soon_threadsafe(shutdown_requested.set)

    try:
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        activity = ActivityCoordinator()
        if backend_type == "gazebo":
            flush_ros_environment()
            if csd_manifest is not None:
                from robosim.backends.gazebo.runtime import GazeboRuntime

                gazebo_runtime = GazeboRuntime(Path(csd_manifest), headless=headless)
                gazebo_runtime.start()
            import rclpy as gazebo_rclpy
            from rclpy.signals import SignalHandlerOptions

            rclpy = gazebo_rclpy
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        backend = create_backend(
            backend_type=backend_type,
            robot_name=robot_name,
            scene=scene,
            csd_manifest=csd_manifest,
            headless=headless,
        )

        recorder = LerobotDataRecorder(DATA_REPO_ROOT, backend, activity_coordinator=activity)
        policy_runner = LerobotPolicyRunner(
            DATA_REPO_ROOT,
            backend,
            activity_coordinator=activity,
        )
        server = create_server(backend, recorder, policy_runner, port)

        await server.start()
        print(f"gRPC server started on port {port}")
        print(f"Backend: {backend_type}")
        print(f"Robot: {backend.robot_name}")
        print(f"Capabilities: {backend.capabilities}")
        print("Press Ctrl+C to stop")

        await shutdown_requested.wait()
    except Exception as e:
        print(f"Server error: {e}")
        raise
    finally:
        await shutdown()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse server command-line arguments."""
    parser = argparse.ArgumentParser(description="RoboSim gRPC Server")
    parser.add_argument(
        "--backend",
        type=str,
        default="gazebo",
        choices=["gazebo", "mujoco", "pybullet"],
        help="Simulator backend type",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="robot",
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
        help="Path to a backend-native scene file",
    )
    parser.add_argument(
        "--csd-manifest",
        type=str,
        default=None,
        help="Path to a compiled CSD realization manifest.json",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run supported backends without viewer",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Start the gRPC server from command-line arguments."""
    args = parse_args()

    asyncio.run(serve_async(
        backend_type=args.backend,
        robot_name=args.robot_name,
        port=args.port,
        scene=args.scene,
        csd_manifest=args.csd_manifest,
        headless=args.headless,
    ))


if __name__ == "__main__":
    main()
