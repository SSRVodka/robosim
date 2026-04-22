"""SensingService gRPC implementation."""

from __future__ import annotations

from collections.abc import Iterator

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import sensing_pb2 as sensing_pb2
from control_stubs import sensing_pb2_grpc as sensing_pb2_grpc
from robosim.core.backend import SimulatorBackend


class SensingServicer(sensing_pb2_grpc.SensingServiceServicer):
    """gRPC servicer for sensor data."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def ListSensors(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> sensing_pb2.SensorMetaList:
        return self._backend.list_sensors()

    def GetSensors(
        self, request: sensing_pb2.SensorRequest, context: grpc.ServicerContext
    ) -> sensing_pb2.SensorData:
        return self._backend.get_sensors(list(request.sensor_names))

    def StreamSensors(
        self, request: sensing_pb2.SensorRequest, context: grpc.ServicerContext
    ) -> Iterator[sensing_pb2.SensorData]:
        try:
            yield from self._backend.stream_sensors(list(request.sensor_names))
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            yield sensing_pb2.SensorData()
