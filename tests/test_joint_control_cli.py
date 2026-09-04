from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_stubs import common_pb2, robot_core_pb2
from control_stubs.tools import joint_control


class FakeRobotCore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[float], int, str]] = []

    def get_robot_spec(self) -> robot_core_pb2.RobotSpecification:
        return robot_core_pb2.RobotSpecification(
            robot_name="robot",
            joint_model_groups=[
                robot_core_pb2.JointModelGroupSpec(name="arm", joint_names=["j1", "j2"])
            ],
        )

    def set_joint_target(
        self, names: list[str], data: list[float], mode: int, jmg_name: str
    ) -> common_pb2.Status:
        self.calls.append((names, data, mode, jmg_name))
        return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)

    def get_robot_state(self) -> common_pb2.JointState:
        return common_pb2.JointState(name=["j1", "j2"], position=[0.1, -0.2])


def test_joint_control_resolves_group_and_waits_for_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    robot_core = FakeRobotCore()
    client = SimpleNamespace(robot_core=robot_core, close=lambda: None)
    monkeypatch.setattr(joint_control, "RobosimClient", lambda host, port: client)

    result = joint_control.main(["--group", "arm", "--positions", "0.1,-0.2"])

    assert result == 0
    assert robot_core.calls == [
        (["j1", "j2"], [0.1, -0.2], robot_core_pb2.JointCommand.POSITION, "arm")
    ]
    assert "converged: max_error=0" in capsys.readouterr().out


def test_joint_control_rejects_wrong_position_count(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SimpleNamespace(robot_core=FakeRobotCore(), close=lambda: None)
    monkeypatch.setattr(joint_control, "RobosimClient", lambda host, port: client)

    with pytest.raises(SystemExit, match="2"):
        joint_control.main(["--group", "arm", "--positions", "0.1"])
