"""gRPC client for robosim simulator control."""

from __future__ import annotations

from collections.abc import Iterator

import grpc

from control_stubs import (
    common_pb2,
    mobility_ai_pb2,
    mobility_ai_pb2_grpc,
    robot_core_pb2,
    robot_core_pb2_grpc,
    robot_data_pb2,
    robot_data_pb2_grpc,
    sensing_pb2,
    sensing_pb2_grpc,
    simulation_pb2,
    simulation_pb2_grpc,
)


class RobosimClient:
    """Unified gRPC client for all simulator control services."""

    def __init__(self, host: str = "localhost", port: int = 50051) -> None:
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self.simulation = SimulationStub(self._channel)
        self.sensing = SensingStub(self._channel)
        self.robot_core = RobotCoreStub(self._channel)
        self.robot_data = RobotDataStub(self._channel)
        self.mobility = MobilityStub(self._channel)

    def close(self) -> None:
        self._channel.close()


class SimulationStub:
    """SimulationService client stub."""

    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = simulation_pb2_grpc.SimulationServiceStub(channel)

    def reset_world(
        self, seed: int = 0, randomization_params: dict[str, float] | None = None
    ) -> common_pb2.Status:
        req = simulation_pb2.ResetRequest(
            seed=seed,
            randomization_params=randomization_params or {},
        )
        return self._stub.ResetWorld(req)

    def step_physics(self) -> simulation_pb2.StepResponse:
        return self._stub.StepPhysics()

    def set_object_pose(
        self,
        object_name: str,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> common_pb2.Status:
        pose = common_pb2.Pose(
            position=common_pb2.Point(x=position[0], y=position[1], z=position[2]),
            orientation=common_pb2.Quaternion(
                x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
            ),
        )
        req = simulation_pb2.ObjectState(object_name=object_name, pose=pose)
        return self._stub.SetObjectPose(req)


class SensingStub:
    """SensingService client stub."""

    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = sensing_pb2_grpc.SensingServiceStub(channel)

    def list_sensors(self) -> sensing_pb2.SensorMetaList:
        return self._stub.ListSensors(common_pb2.Empty())

    def get_sensors(self, sensor_names: list[str]) -> sensing_pb2.SensorData:
        req = sensing_pb2.SensorRequest(sensor_names=sensor_names)
        return self._stub.GetSensors(req)

    def stream_sensors(self, sensor_names: list[str]) -> Iterator[sensing_pb2.SensorData]:
        req = sensing_pb2.SensorRequest(sensor_names=sensor_names)
        return self._stub.StreamSensors(req)


class RobotCoreStub:
    """RobotCoreService client stub."""

    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = robot_core_pb2_grpc.RobotCoreServiceStub(channel)

    def get_robot_state(self) -> common_pb2.JointState:
        return self._stub.GetRobotState(common_pb2.Empty())

    def get_robot_spec(self) -> robot_core_pb2.RobotSpecification:
        return self._stub.GetRobotSpec(common_pb2.Empty())

    def set_joint_target(
        self,
        names: list[str],
        data: list[float],
        mode: robot_core_pb2.JointCommand.ControlMode,
        jmg_name: str | None = None,
    ) -> common_pb2.Status:
        group = robot_core_pb2.JointModelGroupRequest(jmg_name=jmg_name) if jmg_name else None
        req = robot_core_pb2.JointCommand(name=names, data=data, mode=mode, group=group)
        return self._stub.SetJointTarget(req)

    def get_end_effector_state(self, jmg_name: str) -> robot_core_pb2.EndEffectorState:
        req = robot_core_pb2.JointModelGroupRequest(jmg_name=jmg_name)
        return self._stub.GetEndEffectorState(req)

    def servo_control_stream(
        self, commands: Iterator[robot_core_pb2.ServoCommand]
    ) -> Iterator[common_pb2.JointState]:
        return self._stub.ServoControlStream(commands)

    def emergency_stop(self) -> common_pb2.Status:
        return self._stub.EmergencyStop(common_pb2.Empty())


class MobilityStub:
    """MobilityService client stub."""

    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = mobility_ai_pb2_grpc.MobilityServiceStub(channel)

    def get_robot_pose_in_map(self) -> common_pb2.PoseStamped:
        return self._stub.GetRobotPoseInMap(common_pb2.Empty())

    def navigate_to(
        self,
        target_pose: tuple[float, float, float, float, float, float],
        target_frame: str = "map",
        max_velocity: float = 0.0,
    ) -> Iterator[mobility_ai_pb2.TaskFeedback]:
        px, py, pz, qx, qy, qz = target_pose
        pose = common_pb2.Pose(
            position=common_pb2.Point(x=px, y=py, z=pz),
            orientation=common_pb2.Quaternion(x=qx, y=qy, z=qz, w=1.0),
        )
        req = mobility_ai_pb2.NavGoal(
            target_pose=pose,
            target_frame=target_frame,
            max_velocity=max_velocity,
        )
        return self._stub.NavigateTo(req)


class RobotDataStub:
    """RobotDataService client stub."""

    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = robot_data_pb2_grpc.RobotDataServiceStub(channel)

    def record_episode_start(
        self,
        repo_name: str,
        task_text: str = "",
        fps: int = 0,
        jmg_included: list[str] | None = None,
        jmg_excluded: list[str] | None = None,
        sensor_name_included: list[str] | None = None,
        sensor_name_excluded: list[str] | None = None,
    ) -> robot_data_pb2.RecordJobInfo:
        req = robot_data_pb2.RecordOptions(
            repo_name=repo_name,
            task_text=task_text,
            fps=fps,
            jmg_included=jmg_included or [],
            jmg_excluded=jmg_excluded or [],
            sensor_name_included=sensor_name_included or [],
            sensor_name_excluded=sensor_name_excluded or [],
        )
        return self._stub.RecordEpisodeStart(req)

    def record_episode_end(self) -> common_pb2.Status:
        return self._stub.RecordEpisodeEnd(common_pb2.Empty())
