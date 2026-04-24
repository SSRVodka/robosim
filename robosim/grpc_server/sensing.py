"""SensingService gRPC implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import sensing_pb2 as sensing_pb2
from control_stubs import sensing_pb2_grpc as sensing_pb2_grpc
from robosim.core.backend import SimulatorBackend

_logger = logging.getLogger(__name__)


class SensingServicer(sensing_pb2_grpc.SensingServiceServicer):
    """gRPC servicer for sensor data."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def ListSensors(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> sensing_pb2.SensorMetaList:
        _logger.info("ListSensors called")
        try:
            result = self._backend.list_sensors()
            _logger.info("ListSensors succeeded: %d sensors", len(result.sensors))
            return result
        except Exception as e:
            _logger.error("ListSensors failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return sensing_pb2.SensorMetaList()

    def GetSensors(
        self, request: sensing_pb2.SensorRequest, context: grpc.ServicerContext
    ) -> sensing_pb2.SensorData:
        sensor_names = list(request.sensor_names)
        _logger.info("GetSensors called: sensor_names=%s", sensor_names)
        try:
            result = self._backend.get_sensors(sensor_names)
            _logger.info("GetSensors succeeded: %d sensor readings", len(result.data))
            return result
        except NotImplementedError:
            _logger.warning("GetSensors not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return sensing_pb2.SensorData()
        except Exception as e:
            _logger.error("GetSensors failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return sensing_pb2.SensorData()

    def StreamSensors(
        self, request: sensing_pb2.SensorRequest, context: grpc.ServicerContext
    ) -> Iterator[sensing_pb2.SensorData]:
        sensor_names = list(request.sensor_names)
        _logger.info("StreamSensors started: sensor_names=%s", sensor_names)
        try:
            yield from self._backend.stream_sensors(sensor_names)
            _logger.info("StreamSensors completed")
        except NotImplementedError:
            _logger.warning("StreamSensors not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return
        except Exception as e:
            _logger.error("StreamSensors failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return
