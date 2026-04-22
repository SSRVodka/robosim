"""SimulationService gRPC implementation."""

from __future__ import annotations

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import simulation_pb2 as sim_pb2
from control_stubs import simulation_pb2_grpc as sim_pb2_grpc
from robosim.core.backend import SimulatorBackend


class SimulationServicer(sim_pb2_grpc.SimulationServiceServicer):
    """gRPC servicer for simulation control."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self._backend = backend

    def ResetWorld(
        self, request: sim_pb2.ResetRequest, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        try:
            self._backend.reset_world(request.seed, dict(request.randomization_params))
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported by backend"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))

    def StepPhysics(
        self, request: sim_pb2.StepRequest, context: grpc.ServicerContext
    ) -> sim_pb2.StepResponse:
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return sim_pb2.StepResponse(
            header=common_pb2.Header(seq=0, timestamp=0.0, frame_id=""),
            reward=0.0,
            done=False,
        )

    def SetObjectPose(
        self, request: sim_pb2.ObjectState, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return common_pb2.Status(
            code=common_pb2.STATUS_FAILURE, message="Not implemented"
        )
