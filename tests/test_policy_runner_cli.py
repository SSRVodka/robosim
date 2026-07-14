from __future__ import annotations

import pytest

from control_stubs import common_pb2, policy_pb2
from control_stubs.tools import policy_runner


class FakeChannel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_policy_runner_cli_load_invokes_policy_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    channel = FakeChannel()

    class FakeStub:
        def __init__(self, stub_channel: FakeChannel) -> None:
            calls["channel"] = stub_channel

        def LoadPolicy(self, request: policy_pb2.PolicyLoadRequest) -> common_pb2.Status:
            calls["load"] = request
            return common_pb2.Status(
                code=common_pb2.STATUS_SUCCESS,
                message="policy loaded",
            )

    monkeypatch.setattr(policy_runner.grpc, "insecure_channel", lambda target: channel)
    monkeypatch.setattr(policy_runner.policy_pb2_grpc, "PolicyInferenceServiceStub", FakeStub)

    exit_code = policy_runner.main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "6000",
            "load",
            "--policy-path",
            "/tmp/policy",
            "--dataset-repo-name",
            "panda_policy_dataset",
            "--device",
            "cpu",
            "--task-text",
            "hold pose",
            "--jmg-name",
            "panda_arm",
            "--control-fps",
            "5",
        ]
    )

    assert exit_code == 0
    assert channel.closed
    request = calls["load"]
    assert isinstance(request, policy_pb2.PolicyLoadRequest)
    assert request.policy_path == "/tmp/policy"
    assert request.dataset_repo_name == "panda_policy_dataset"
    assert request.device == "cpu"
    assert request.task_text == "hold pose"
    assert request.jmg_name == "panda_arm"
    assert request.control_fps == 5


def test_policy_runner_cli_start_and_stop_invoke_policy_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    channels: list[FakeChannel] = []

    class FakeStub:
        def __init__(self, stub_channel: FakeChannel) -> None:
            calls["channel"] = stub_channel

        def StartPolicy(self, request: policy_pb2.PolicyStartRequest) -> common_pb2.Status:
            calls["start"] = request
            return common_pb2.Status(
                code=common_pb2.STATUS_SUCCESS,
                message="policy started",
            )

        def StopPolicy(self, request: common_pb2.Empty) -> common_pb2.Status:
            calls["stop"] = request
            return common_pb2.Status(
                code=common_pb2.STATUS_SUCCESS,
                message="policy stopped",
            )

    def create_channel(target: str) -> FakeChannel:
        del target
        channel = FakeChannel()
        channels.append(channel)
        return channel

    monkeypatch.setattr(policy_runner.grpc, "insecure_channel", create_channel)
    monkeypatch.setattr(policy_runner.policy_pb2_grpc, "PolicyInferenceServiceStub", FakeStub)

    start_exit_code = policy_runner.main(
        ["start", "--task-text", "hold pose", "--control-fps", "5"]
    )
    stop_exit_code = policy_runner.main(["stop"])

    assert start_exit_code == 0
    assert stop_exit_code == 0
    start_request = calls["start"]
    assert isinstance(start_request, policy_pb2.PolicyStartRequest)
    assert start_request.task_text == "hold pose"
    assert start_request.control_fps == 5
    assert isinstance(calls["stop"], common_pb2.Empty)
    assert [channel.closed for channel in channels] == [True, True]


def test_policy_runner_cli_status_prints_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    channel = FakeChannel()

    class FakeStub:
        def __init__(self, stub_channel: FakeChannel) -> None:
            del stub_channel

        def GetPolicyStatus(self, request: common_pb2.Empty) -> policy_pb2.PolicyStatus:
            del request
            return policy_pb2.PolicyStatus(
                status=common_pb2.Status(code=common_pb2.STATUS_RUNNING),
                loaded=True,
                running=True,
                policy_type="act",
                policy_path="/tmp/policy",
                dataset_repo_name="panda_policy_dataset",
                device="cpu",
                jmg_name="panda_arm",
                control_fps=5,
                task_text="hold pose",
                active_mode="policy",
            )

    monkeypatch.setattr(policy_runner.grpc, "insecure_channel", lambda target: channel)
    monkeypatch.setattr(policy_runner.policy_pb2_grpc, "PolicyInferenceServiceStub", FakeStub)

    exit_code = policy_runner.main(["status"])

    assert exit_code == 0
    assert channel.closed
    out = capsys.readouterr().out
    assert "loaded: true" in out
    assert "running: true" in out
    assert "policy_type: act" in out
    assert "jmg_name: panda_arm" in out
