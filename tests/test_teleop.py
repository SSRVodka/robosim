from __future__ import annotations

import pytest

from control_stubs import common_pb2, robot_core_pb2, robot_data_pb2
from control_stubs.tools import teleop
from control_stubs.tools.joycon import JoyConInput
from control_stubs.tools.teleop import (
    EpisodeController,
    InputSnapshot,
    RecordingConfig,
    ServoSession,
    TeleopEvent,
    TeleopMotion,
    build_parser,
    build_target_catalog,
)


def _spec() -> robot_core_pb2.RobotSpecification:
    return robot_core_pb2.RobotSpecification(
        robot_name="demo",
        joint_model_groups=[
            robot_core_pb2.JointModelGroupSpec(
                name="left_arm",
                joint_names=["left_joint"],
                end_effectors=[
                    robot_core_pb2.EESpec(name="left_tool", parent_jmg_name="left_arm")
                ],
            ),
            robot_core_pb2.JointModelGroupSpec(
                name="right_arm",
                joint_names=["right_joint"],
                end_effectors=[
                    robot_core_pb2.EESpec(name="right_tool", parent_jmg_name="right_arm")
                ],
            ),
            robot_core_pb2.JointModelGroupSpec(name="left_hand", joint_names=["left_finger"]),
            robot_core_pb2.JointModelGroupSpec(name="right_hand", joint_names=["right_finger"]),
        ],
    )


def test_servo_session_stops_old_targets_before_sending_to_new_targets() -> None:
    commands: list[robot_core_pb2.ServoCommand] = []
    targets = build_target_catalog(
        _spec(),
        twist_targets=["left_arm:left_tool", "right_arm:right_tool"],
        joint_targets=["left_hand", "right_hand"],
    )
    session = ServoSession(targets, commands.append)

    remaining = session.apply(
        InputSnapshot(
            motion=TeleopMotion(linear=(0.02, 0.0, 0.0), joint_velocity=0.2),
            events=(TeleopEvent.NEXT_TWIST_TARGET, TeleopEvent.NEXT_JOINT_TARGET),
        )
    )

    assert remaining == ()
    assert commands[0].twist_cmd.target_ee.parent_jmg_name == "left_arm"
    assert commands[0].twist_cmd.twist.twist.linear.x == 0.0
    assert commands[1].joint_cmd.group.jmg_name == "left_hand"
    assert list(commands[1].joint_cmd.data) == [0.0]
    assert commands[2].twist_cmd.target_ee.parent_jmg_name == "right_arm"
    assert commands[2].twist_cmd.twist.twist.linear.x == 0.02
    assert commands[3].joint_cmd.group.jmg_name == "right_hand"
    assert list(commands[3].joint_cmd.data) == [0.2]


def test_servo_session_zeroes_motion_before_episode_event() -> None:
    commands: list[robot_core_pb2.ServoCommand] = []
    session = ServoSession(
        build_target_catalog(
            _spec(),
            twist_targets=["left_arm:left_tool"],
            joint_targets=["left_hand"],
        ),
        commands.append,
    )
    motion = TeleopMotion(linear=(0.02, 0.0, 0.0), joint_velocity=0.2)
    session.apply(InputSnapshot(motion=motion))
    commands.clear()

    remaining = session.apply(
        InputSnapshot(motion=motion, events=(TeleopEvent.SAVE_EPISODE,))
    )

    assert remaining == (TeleopEvent.SAVE_EPISODE,)
    assert commands[0].twist_cmd.twist.twist.linear.x == 0.0
    assert list(commands[1].joint_cmd.data) == [0.0]


class FakeRobotData:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def episode_start(self, **kwargs) -> robot_data_pb2.RecordJobInfo:
        self.calls.append(f"start:{kwargs['repo_name']}:{kwargs['jmg_included']}")
        return robot_data_pb2.RecordJobInfo(
            status=common_pb2.Status(code=common_pb2.STATUS_SUCCESS)
        )

    def episode_end(self) -> common_pb2.Status:
        self.calls.append("end")
        return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)

    def episode_cancel(self) -> common_pb2.Status:
        self.calls.append("cancel")
        return common_pb2.Status(code=common_pb2.STATUS_SUCCESS)


class FakeSimulation:
    def __init__(self, calls: list[str], *, succeeds: bool = True) -> None:
        self.calls = calls
        self.succeeds = succeeds

    def reset_world(self) -> common_pb2.Status:
        self.calls.append("reset")
        return common_pb2.Status(
            code=(
                common_pb2.STATUS_SUCCESS
                if self.succeeds
                else common_pb2.STATUS_FAILURE
            ),
            message="reset unavailable",
        )


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.robot_data = FakeRobotData(self.calls)
        self.simulation = FakeSimulation(self.calls)


def test_episode_controller_orders_save_retry_reset_and_next_start() -> None:
    client = FakeClient()
    controller = EpisodeController(
        client,  # type: ignore[arg-type]
        RecordingConfig(
            repo_name="quick_dataset",
            task_text="pick object",
            fps=30,
            jmg_names=("left_arm", "left_hand"),
            sensor_names=(),
            reset_between_episodes=True,
        ),
    )

    controller.start()
    controller.handle(TeleopEvent.SAVE_EPISODE)
    controller.handle(TeleopEvent.RETRY_EPISODE)
    controller.stop()

    assert client.calls == [
        "start:quick_dataset:['left_arm', 'left_hand']",
        "end",
        "reset",
        "start:quick_dataset:['left_arm', 'left_hand']",
        "cancel",
        "reset",
        "start:quick_dataset:['left_arm', 'left_hand']",
        "cancel",
    ]


def test_episode_controller_skips_reset_unless_enabled() -> None:
    client = FakeClient()
    controller = EpisodeController(
        client,  # type: ignore[arg-type]
        RecordingConfig(
            repo_name="quick_dataset",
            task_text="pick object",
            fps=30,
            jmg_names=("left_arm", "left_hand"),
            sensor_names=(),
        ),
    )

    controller.start()
    controller.handle(TeleopEvent.SAVE_EPISODE)
    controller.stop()

    assert client.calls == [
        "start:quick_dataset:['left_arm', 'left_hand']",
        "end",
        "start:quick_dataset:['left_arm', 'left_hand']",
        "cancel",
    ]


def test_episode_controller_does_not_start_after_reset_failure() -> None:
    client = FakeClient()
    client.simulation = FakeSimulation(client.calls, succeeds=False)
    controller = EpisodeController(
        client,  # type: ignore[arg-type]
        RecordingConfig(
            repo_name="quick_dataset",
            task_text="pick object",
            fps=30,
            jmg_names=("left_arm", "left_hand"),
            sensor_names=(),
            reset_between_episodes=True,
        ),
    )
    controller.start()

    with pytest.raises(RuntimeError, match="failed to reset world"):
        controller.handle(TeleopEvent.RETRY_EPISODE)
    controller.stop()

    assert client.calls == [
        "start:quick_dataset:['left_arm', 'left_hand']",
        "cancel",
        "reset",
    ]


def test_teleop_parser_enables_parameterized_joycon_collection() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "joycon",
            "--input-device",
            "/dev/input/event15",
            "--input-profile",
            "joycon-right",
            "--twist-target",
            "panda_arm:hand",
            "--joint-target",
            "panda_hand",
            "--repo-name",
            "demo1",
            "--task-text",
            "pick and place",
            "--fps",
            "30",
            "--reset-between-episodes",
        ]
    )

    assert args.input == "joycon"
    assert args.input_device == "/dev/input/event15"
    assert args.twist_target == ["panda_arm:hand"]
    assert args.joint_target == ["panda_hand"]
    assert args.repo_name == "demo1"
    assert args.reset_between_episodes


class ClosingInput:
    fd = 42

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close(self) -> None:
        self.calls.append("input_close")


class ClosingRobotCore:
    def get_robot_spec(self) -> robot_core_pb2.RobotSpecification:
        return _spec()

    def servo_control_stream(self, commands):
        del commands
        return iter(())


class FailingRobotData(FakeRobotData):
    def episode_cancel(self) -> common_pb2.Status:
        self.calls.append("cancel")
        raise RuntimeError("cancel failed")


class ClosingClient(FakeClient):
    def __init__(self, *, cancel_fails: bool = False) -> None:
        super().__init__()
        self.robot_core = ClosingRobotCore()
        if cancel_fails:
            self.robot_data = FailingRobotData(self.calls)

    def close(self) -> None:
        self.calls.append("client_close")


def test_joycon_run_closes_device_and_client_when_episode_cancel_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ClosingClient(cancel_fails=True)
    input_device = ClosingInput(client.calls)
    monkeypatch.setattr(teleop, "RobosimClient", lambda *args: client)
    monkeypatch.setattr(JoyConInput, "open", lambda *args, **kwargs: input_device)
    args = build_parser().parse_args(
        [
            "--input",
            "joycon",
            "--input-device",
            "/dev/input/fake",
            "--twist-target",
            "left_arm:left_tool",
            "--joint-target",
            "left_hand",
            "--repo-name",
            "quick_dataset",
        ]
    )

    with pytest.raises(RuntimeError, match="cancel failed"):
        teleop.run(args)

    assert "input_close" in client.calls
    assert "client_close" in client.calls


def test_joycon_run_closes_client_when_device_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ClosingClient()
    monkeypatch.setattr(teleop, "RobosimClient", lambda *args: client)

    def fail_open(*args, **kwargs):
        raise OSError("device unavailable")

    monkeypatch.setattr(JoyConInput, "open", fail_open)
    args = build_parser().parse_args(
        ["--input", "joycon", "--input-device", "/dev/input/missing"]
    )

    with pytest.raises(OSError, match="device unavailable"):
        teleop.run(args)

    assert "client_close" in client.calls
