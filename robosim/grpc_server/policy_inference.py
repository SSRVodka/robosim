"""PolicyInferenceService gRPC implementation."""

from __future__ import annotations

import logging

import grpc

from control_stubs import common_pb2, policy_pb2, policy_pb2_grpc
from robosim.core.impl.policy_lerobot import LerobotPolicyRunner

_logger = logging.getLogger(__name__)


class PolicyInferenceServicer(policy_pb2_grpc.PolicyInferenceServiceServicer):
    def __init__(self, runner: LerobotPolicyRunner) -> None:
        self._runner = runner

    def LoadPolicy(
        self,
        request: policy_pb2.PolicyLoadRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Status:
        try:
            return self._runner.load_policy(request)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message="Not supported")
        except Exception as exc:
            _logger.error("LoadPolicy failed: %s", exc, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(exc))

    def StartPolicy(
        self,
        request: policy_pb2.PolicyStartRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Status:
        try:
            return self._runner.start_policy(request)
        except NotImplementedError:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message="Not supported")
        except Exception as exc:
            _logger.error("StartPolicy failed: %s", exc, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(exc))

    def StopPolicy(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.Status:
        del request
        try:
            return self._runner.stop_policy()
        except Exception as exc:
            _logger.error("StopPolicy failed: %s", exc, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            return common_pb2.Status(code=common_pb2.STATUS_FAILURE, message=str(exc))

    def GetPolicyStatus(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> policy_pb2.PolicyStatus:
        del request, context
        return self._runner.get_status()
