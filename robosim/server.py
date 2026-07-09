#!/usr/bin/env python3
"""gRPC server main entry point."""

from __future__ import annotations

import argparse
import asyncio
import signal
from concurrent import futures
from pathlib import Path

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
from robosim.backends import GazeboBackend, MuJoCoBackend
from robosim.core.activity import ActivityCoordinator
from robosim.core.backend import SimulatorBackend
from robosim.core.impl.policy_lerobot import LerobotPolicyRunner
from robosim.core.impl.recorder_lerobot import LerobotDataRecorder
from robosim.grpc_server import (
    MobilityServicer,
    PolicyInferenceServicer,
    RobotCoreServicer,
    RobotDataServicer,
    SensingServicer,
    SimulationServicer,
)

DATA_REPO_ROOT = Path(__file__).resolve().parent.parent


def create_backend(
    *,
    backend_type: str,
    robot_name: str,
    scene: str | None,
    csd_manifest: str | None,
    headless: bool,
) -> SimulatorBackend:
    """Create a simulator backend for server startup."""
    if backend_type == "gazebo":
        return GazeboBackend(robot_name=robot_name)
    if backend_type == "mujoco":
        if csd_manifest is not None:
            return MuJoCoBackend.from_csd_realization_manifest_file(
                Path(csd_manifest),
                headless=headless,
            )
        return MuJoCoBackend(
            scene_path=scene or "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml",
            headless=headless,
        )
    raise ValueError(f"Unknown backend type: {backend_type}")


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
    import rclpy

    backend: SimulatorBackend | None = None
    recorder: LerobotDataRecorder | None = None
    policy_runner: LerobotPolicyRunner | None = None
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
            rclpy.init()
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

        while not should_shutdown:
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Server error: {e}")
        if policy_runner is not None:
            policy_runner.shutdown()
        if backend is not None:
            backend.shutdown()
        try:
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
        choices=["gazebo", "mujoco"],
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
        help="Path to MuJoCo scene XML file (for mujoco backend)",
    )
    parser.add_argument(
        "--csd-manifest",
        type=str,
        default=None,
        help="Path to a compiled CSD realization manifest.json (for mujoco backend)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MuJoCo without viewer",
    )
    args = parser.parse_args()

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
