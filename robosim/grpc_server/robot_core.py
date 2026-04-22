"""RobotCoreService gRPC implementation."""

from __future__ import annotations

from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs import robot_core_pb2_grpc as core_pb2_grpc
from robosim.core.backend import SimulatorBackend


class RobotCoreServicer(core_pb2_grpc.RobotCoreServiceServicer):
    """gRPC servicer for robot core operations."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def GetRobotState(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.JointState:
        return self._backend.get_robot_state()

    def GetRobotSpec(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> core_pb2.RobotSpecification:
        try:
            return self._backend.get_robot_spec()
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return core_pb2.RobotSpecification()

    def SetJointTarget(
        self, request: core_pb2.JointCommand, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        try:
            self._backend.set_joint_target(
                list(request.name),
                list(request.data),
                core_pb2.JointCommand.ControlMode(request.mode),
                request.group.jmg_name if request.HasField("group") else None,
            )
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))

    def GetEndEffectorState(
        self, request: core_pb2.MoveGroupRequest, context: grpc.ServicerContext
    ) -> core_pb2.EndEffectorState:
        try:
            return self._backend.get_end_effector_state(request.jmg_name)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return core_pb2.EndEffectorState()

    def ServoControlStream(
        self, request_iterator: grpc.RpcMethodHandler, context: grpc.ServicerContext
    ) -> Iterator[common_pb2.JointState]:
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return iter([])

    def EmergencyStop(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        try:
            self._backend.emergency_stop()
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except Exception as e:
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
