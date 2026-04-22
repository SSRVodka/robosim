"""MobilityService gRPC implementation."""

from __future__ import annotations

from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import mobility_ai_pb2 as mobility_pb2
from control_stubs import mobility_ai_pb2_grpc as mobility_pb2_grpc
from robosim.core.backend import SimulatorBackend


class MobilityServicer(mobility_pb2_grpc.MobilityServiceServicer):
    """gRPC servicer for mobility and navigation."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def GetRobotPoseInMap(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.PoseStamped:
        try:
            return self._backend.get_robot_pose_in_map()
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.PoseStamped()

    def NavigateTo(
        self, request: mobility_pb2.NavGoal, context: grpc.ServicerContext
    ) -> Iterator[mobility_pb2.TaskFeedback]:
        try:
            yield from self._backend.navigate_to(request)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            yield mobility_pb2.TaskFeedback(
                task_id="",
                status=common_pb2.Status(
                    code=common_pb2.STATUS_FAILURE, message="Navigation not supported"
                ),
            )
