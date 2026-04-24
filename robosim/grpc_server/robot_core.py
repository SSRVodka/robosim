"""RobotCoreService gRPC implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs import robot_core_pb2_grpc as core_pb2_grpc
from robosim.core.backend import SimulatorBackend

_logger = logging.getLogger(__name__)


class RobotCoreServicer(core_pb2_grpc.RobotCoreServiceServicer):
    """gRPC servicer for robot core operations."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def GetRobotState(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.JointState:
        _logger.info("GetRobotState called")
        try:
            result = self._backend.get_robot_state()
            _logger.info("GetRobotState succeeded: %s joints", len(result.name))
            return result
        except NotImplementedError:
            _logger.warning("GetRobotState not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.JointState()
        except Exception as e:
            _logger.error("GetRobotState failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.JointState()

    def GetRobotSpec(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> core_pb2.RobotSpecification:
        _logger.info("GetRobotSpec called")
        try:
            result = self._backend.get_robot_spec()
            _logger.info("GetRobotSpec succeeded")
            return result
        except NotImplementedError:
            _logger.warning("GetRobotSpec not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return core_pb2.RobotSpecification()
        except Exception as e:
            _logger.error("GetRobotSpec failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return core_pb2.RobotSpecification()

    def SetJointTarget(
        self, request: core_pb2.JointCommand, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info(
            "SetJointTarget called: names=%s, mode=%s, group=%s",
            list(request.name),
            request.mode,
            request.group.jmg_name if request.HasField("group") else None,
        )
        try:
            self._backend.set_joint_target(
                list(request.name),
                list(request.data),
                request.mode,
                request.group.jmg_name if request.HasField("group") else None,
            )
            _logger.info("SetJointTarget succeeded")
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except NotImplementedError:
            _logger.warning("SetJointTarget not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported"
            )
        except Exception as e:
            _logger.error("SetJointTarget failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))

    def GetEndEffectorState(
        self, request: core_pb2.MoveGroupRequest, context: grpc.ServicerContext
    ) -> core_pb2.EndEffectorState:
        _logger.info("GetEndEffectorState called: jmg_name=%s", request.jmg_name)
        try:
            result = self._backend.get_end_effector_state(request.jmg_name)
            _logger.info("GetEndEffectorState succeeded")
            return result
        except NotImplementedError:
            _logger.warning("GetEndEffectorState not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return core_pb2.EndEffectorState()
        except Exception as e:
            _logger.error("GetEndEffectorState failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return core_pb2.EndEffectorState()

    def ServoControlStream(
        self, request_iterator: Iterator[core_pb2.ServoCommand], context: grpc.ServicerContext
    ) -> Iterator[common_pb2.JointState]:
        _logger.info("ServoControlStream started")
        try:
            yield from self._backend.servo_control_stream(request_iterator)
        except NotImplementedError:
            _logger.warning("ServoControlStream not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return
        except Exception as e:
            _logger.error("ServoControlStream failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal servo error: {str(e)}")
            return

    def EmergencyStop(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info("EmergencyStop called")
        try:
            self._backend.emergency_stop()
            _logger.info("EmergencyStop succeeded")
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except Exception as e:
            _logger.error("EmergencyStop failed: %s", e, exc_info=True)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
