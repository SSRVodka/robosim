"""MobilityService gRPC implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import mobility_ai_pb2 as mobility_pb2
from control_stubs import mobility_ai_pb2_grpc as mobility_pb2_grpc
from robosim.core.backend import SimulatorBackend

_logger = logging.getLogger(__name__)


class MobilityServicer(mobility_pb2_grpc.MobilityServiceServicer):
    """gRPC servicer for mobility and navigation."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def GetRobotPoseInMap(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.PoseStamped:
        _logger.info("GetRobotPoseInMap called")
        try:
            result = self._backend.get_robot_pose_in_map()
            _logger.info("GetRobotPoseInMap succeeded")
            return result
        except NotImplementedError:
            _logger.warning("GetRobotPoseInMap not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.PoseStamped()
        except Exception as e:
            _logger.error("GetRobotPoseInMap failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.PoseStamped()

    def NavigateTo(
        self, request: mobility_pb2.NavGoal, context: grpc.ServicerContext
    ) -> Iterator[mobility_pb2.TaskFeedback]:
        _logger.info("NavigateTo called: target=%s", request.target_pose)
        try:
            yield from self._backend.navigate_to(request)
            _logger.info("NavigateTo completed")
        except NotImplementedError:
            _logger.warning("NavigateTo not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            yield mobility_pb2.TaskFeedback(
                task_id="",
                status=common_pb2.Status(
                    code=common_pb2.STATUS_FAILURE, message="Navigation not supported"
                ),
            )
        except Exception as e:
            _logger.error("NavigateTo failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            yield mobility_pb2.TaskFeedback(
                task_id="",
                status=common_pb2.Status(
                    code=common_pb2.STATUS_FAILURE, message=str(e)
                ),
            )
