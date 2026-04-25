"""RobotDataService gRPC implementation."""

from __future__ import annotations

import logging

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import robot_data_pb2 as data_pb2
from control_stubs import robot_data_pb2_grpc as data_pb2_grpc
from robosim.core.recorder import DataRecorder

_logger = logging.getLogger(__name__)


class RobotDataServicer(data_pb2_grpc.RobotDataServiceServicer):
    """gRPC servicer for robot data operations."""

    def __init__(self, recorder: DataRecorder) -> None:
        self._recorder = recorder

    def RecordEpisodeStart(
        self, request: data_pb2.RecordOptions, context: grpc.ServicerContext
    ) -> data_pb2.RecordJobInfo:
        _logger.info("RecordEpisodeStart called")
        try:
            job_info = self._recorder.record_episode_start(request)
            _logger.info("RecordEpisodeStart succeeded")
            return job_info
        except NotImplementedError:
            _logger.warning("RecordEpisodeStart not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return data_pb2.RecordJobInfo(status=common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported"), episode_id=-1)
        except Exception as e:
            _logger.error("RecordEpisodeStart failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return data_pb2.RecordJobInfo(status=common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message=str(e)), episode_id=-1)
    
    def RecordEpisodeEnd(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info("RecordEpisodeEnd called")
        try:
            status = self._recorder.record_episode_end()
            _logger.info("RecordEpisodeEnd succeeded")
            return status
        except NotImplementedError:
            _logger.warning("RecordEpisodeEnd not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message="Not supported")
        except Exception as e:
            _logger.error("RecordEpisodeEnd failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
    