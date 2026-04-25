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

    def EpisodeStart(
        self, request: data_pb2.RecordOptions, context: grpc.ServicerContext
    ) -> data_pb2.RecordJobInfo:
        _logger.info("EpisodeStart called")
        try:
            job_info = self._recorder.episode_start(request)
            _logger.info("EpisodeStart succeeded")
            return job_info
        except NotImplementedError:
            _logger.warning("EpisodeStart not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return data_pb2.RecordJobInfo(status=common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported"), episode_id=-1)
        except Exception as e:
            _logger.error("EpisodeStart failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return data_pb2.RecordJobInfo(status=common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message=str(e)), episode_id=-1)
    
    def EpisodeEnd(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info("EpisodeEnd called")
        try:
            status = self._recorder.episode_end()
            _logger.info("EpisodeEnd succeeded")
            return status
        except NotImplementedError:
            _logger.warning("EpisodeEnd not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message="Not supported")
        except Exception as e:
            _logger.error("EpisodeEnd failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
    
    def EpisodeReplay(
        self, request: data_pb2.RecordInfo, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info("EpisodeReplay called")
        try:
            status = self._recorder.episode_replay(request)
            _logger.info("EpisodeReplay succeeded")
            return status
        except NotImplementedError:
            _logger.warning("EpisodeReplay not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message="Not supported")
        except Exception as e:
            _logger.error("EpisodeReplay failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
