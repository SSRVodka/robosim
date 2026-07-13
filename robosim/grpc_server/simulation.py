"""SimulationService gRPC implementation."""

from __future__ import annotations

import logging

import grpc

from control_stubs import common_pb2 as common_pb2
from control_stubs import simulation_pb2 as sim_pb2
from control_stubs import simulation_pb2_grpc as sim_pb2_grpc
from robosim.core.backend import SimulatorBackend
from robosim.core.impl.policy_lerobot import LerobotPolicyRunner

_logger = logging.getLogger(__name__)


class SimulationServicer(sim_pb2_grpc.SimulationServiceServicer):
    """gRPC servicer for simulation control."""

    def __init__(
        self,
        backend: SimulatorBackend,
        policy_runner: LerobotPolicyRunner | None = None,
    ) -> None:
        self._backend = backend
        self._policy_runner = policy_runner

    def ResetWorld(
        self, request: sim_pb2.ResetRequest, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        _logger.info(
            "ResetWorld called: seed=%s, randomization_params=%s",
            request.seed,
            dict(request.randomization_params),
        )
        try:
            self._backend.reset_world(request.seed, dict(request.randomization_params))
            if self._policy_runner is not None:
                self._policy_runner.notify_world_reset()
            _logger.info("ResetWorld succeeded")
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except NotImplementedError:
            _logger.warning("ResetWorld not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not supported by backend"
            )
        except Exception as e:
            _logger.error("ResetWorld failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))

    def StepPhysics(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> sim_pb2.StepResponse:
        del request
        try:
            self._backend.step_physics()  # type: ignore[attr-defined]
            return sim_pb2.StepResponse(
                header=common_pb2.Header(seq=0, timestamp=0.0, frame_id="world"),
                reward=0.0,
                done=False,
            )
        except AttributeError:
            _logger.warning("StepPhysics not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return sim_pb2.StepResponse(
                header=common_pb2.Header(seq=0, timestamp=0.0, frame_id=""),
                reward=0.0,
                done=False,
            )
        except NotImplementedError:
            _logger.warning("StepPhysics not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return sim_pb2.StepResponse(
                header=common_pb2.Header(seq=0, timestamp=0.0, frame_id=""),
                reward=0.0,
                done=False,
            )
    
    def Pause(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Empty:
        del request
        try:
            self._backend.pause()  # type: ignore[attr-defined]
        except AttributeError:
            _logger.warning("Pause not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        except NotImplementedError:
            _logger.warning("Pause not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return common_pb2.Empty()
    
    def Resume(
        self, request: common_pb2.Empty, context: grpc.ServicerContext
    ) -> common_pb2.Empty:
        del request
        try:
            self._backend.resume()  # type: ignore[attr-defined]
        except AttributeError:
            _logger.warning("Resume not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        except NotImplementedError:
            _logger.warning("Resume not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return common_pb2.Empty()

    def SetObjectPose(
        self, request: sim_pb2.ObjectState, context: grpc.ServicerContext
    ) -> common_pb2.Status:
        try:
            self._backend.set_object_pose(  # type: ignore[attr-defined]
                request.object_name,
                request.pose,
            )
            return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        except AttributeError:
            _logger.warning("SetObjectPose not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not implemented"
            )
        except NotImplementedError:
            _logger.warning("SetObjectPose not implemented")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(
                code=common_pb2.STATUS_FAILURE, message="Not implemented"
            )
        except Exception as e:
            _logger.error("SetObjectPose failed: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(e))
