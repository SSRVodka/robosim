from __future__ import annotations

import pytest

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools import servo_keyboard
from control_stubs.tools.servo_keyboard import (
    MotionState,
    build_joint_command,
    build_twist_command,
    select_servo_bindings,
    update_motion_from_key,
)
from control_stubs.tools.teleop import TeleopEvent


def _make_spec() -> core_pb2.RobotSpecification:
    return core_pb2.RobotSpecification(
        robot_name="demo",
        joint_model_groups=[
            core_pb2.JointModelGroupSpec(
                name="arm",
                joint_names=["joint1", "joint2", "joint3"],
                end_effectors=[
                    core_pb2.EESpec(
                        name="tool0",
                        parent_jmg_name="arm",
                        group_name="tool0",
                        parent_link="wrist",
                    )
                ],
            ),
            core_pb2.JointModelGroupSpec(
                name="gripper",
                joint_names=["finger_left", "finger_right"],
            ),
            core_pb2.JointModelGroupSpec(name="empty"),
        ],
    )


def test_select_servo_bindings_prefers_ee_group_and_small_joint_group() -> None:
    bindings = select_servo_bindings(_make_spec())

    assert bindings.twist_group_name == "arm"
    assert bindings.target_ee is not None
    assert bindings.target_ee.name == "tool0"
    assert bindings.joint_group_name == "gripper"
    assert bindings.joint_names == ("finger_left", "finger_right")
    assert bindings.summary_names == ("finger_left", "finger_right")


def test_select_servo_bindings_prefers_second_ee_group_over_shorter_body_group() -> None:
    spec = core_pb2.RobotSpecification(
        robot_name="g1",
        joint_model_groups=[
            core_pb2.JointModelGroupSpec(
                name="left_arm",
                joint_names=["left_shoulder", "left_elbow", "left_wrist"],
                end_effectors=[core_pb2.EESpec(name="left_wrist", parent_jmg_name="left_arm")],
            ),
            core_pb2.JointModelGroupSpec(
                name="waist",
                joint_names=["waist_yaw"],
            ),
            core_pb2.JointModelGroupSpec(
                name="right_arm",
                joint_names=["right_shoulder", "right_elbow", "right_wrist"],
                end_effectors=[core_pb2.EESpec(name="right_wrist", parent_jmg_name="right_arm")],
            ),
        ],
    )

    bindings = select_servo_bindings(spec)

    assert bindings.twist_group_name == "left_arm"
    assert bindings.joint_group_name == "right_arm"


def test_select_servo_bindings_rejects_invalid_explicit_group() -> None:
    with pytest.raises(ValueError, match="has no end effector"):
        select_servo_bindings(_make_spec(), twist_group_name="gripper")


def test_target_catalog_cycles_groups_and_stops_previous_target() -> None:
    spec = core_pb2.RobotSpecification(
        robot_name="dual_arm",
        joint_model_groups=[
            core_pb2.JointModelGroupSpec(
                name="left_arm",
                joint_names=["left_1", "left_2"],
                end_effectors=[
                    core_pb2.EESpec(name="left_tool", parent_jmg_name="left_arm")
                ],
            ),
            core_pb2.JointModelGroupSpec(
                name="right_arm",
                joint_names=["right_1", "right_2"],
                end_effectors=[
                    core_pb2.EESpec(name="right_tool", parent_jmg_name="right_arm")
                ],
            ),
            core_pb2.JointModelGroupSpec(name="left_gripper", joint_names=["left_finger"]),
            core_pb2.JointModelGroupSpec(name="right_gripper", joint_names=["right_finger"]),
        ],
    )
    targets = servo_keyboard.build_target_catalog(
        spec,
        twist_targets=["left_arm:left_tool", "right_arm:right_tool"],
        joint_targets=["left_gripper", "right_gripper"],
    )

    twist_stop = targets.cycle_twist()
    joint_stop = targets.cycle_joint()

    assert targets.active_twist is not None
    assert targets.active_joint is not None
    assert twist_stop is not None
    assert joint_stop is not None
    assert targets.active_twist.group_name == "right_arm"
    assert targets.active_joint.group_name == "right_gripper"
    assert twist_stop.twist_cmd.target_ee.parent_jmg_name == "left_arm"
    assert twist_stop.twist_cmd.twist.twist.linear.x == 0.0
    assert joint_stop.joint_cmd.group.jmg_name == "left_gripper"
    assert list(joint_stop.joint_cmd.data) == [0.0]


def test_keyboard_target_switch_keys_are_independent() -> None:
    targets = servo_keyboard.build_target_catalog(
        _make_spec(),
        twist_targets=["arm:tool0", "arm:tool0"],
        joint_targets=["gripper", "gripper"],
    )

    twist_stop = servo_keyboard.switch_target_from_key("n", targets)
    joint_stop = servo_keyboard.switch_target_from_key("m", targets)

    assert twist_stop is not None and twist_stop.HasField("twist_cmd")
    assert joint_stop is not None and joint_stop.HasField("joint_cmd")
    assert servo_keyboard.switch_target_from_key("x", targets) is None


def test_keyboard_parser_accepts_repeatable_targets() -> None:
    args = servo_keyboard.build_parser().parse_args(
        [
            "--twist-target",
            "left_arm:left_tool",
            "--twist-target",
            "right_arm:right_tool",
            "--joint-target",
            "left_gripper",
            "--joint-target",
            "right_gripper",
        ]
    )

    assert args.twist_target == ["left_arm:left_tool", "right_arm:right_tool"]
    assert args.joint_target == ["left_gripper", "right_gripper"]


def test_keyboard_episode_keys_map_to_device_neutral_events() -> None:
    assert servo_keyboard.episode_event_from_key("e") is TeleopEvent.SAVE_EPISODE
    assert servo_keyboard.episode_event_from_key("c") is TeleopEvent.RETRY_EPISODE
    assert servo_keyboard.episode_event_from_key("q") is TeleopEvent.STOP
    assert servo_keyboard.episode_event_from_key("w") is None


def test_motion_mapping_and_command_building() -> None:
    bindings = select_servo_bindings(_make_spec())
    assert bindings.target_ee is not None
    assert bindings.joint_group_name is not None

    motion = MotionState()
    assert update_motion_from_key(
        motion,
        "w",
        bindings,
        linear_step=0.02,
        angular_step=0.3,
        joint_step=0.2,
        hold_until=1.0,
    )
    assert motion.current_twist(0.5) == ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0))

    twist_command = build_twist_command(bindings.target_ee, *motion.current_twist(0.5))
    assert twist_command.HasField("twist_cmd")
    assert twist_command.twist_cmd.target_ee.name == "tool0"
    assert twist_command.twist_cmd.twist.twist.linear.x == pytest.approx(0.02)

    assert update_motion_from_key(
        motion,
        "]",
        bindings,
        linear_step=0.02,
        angular_step=0.3,
        joint_step=0.2,
        hold_until=2.0,
    )
    assert motion.current_joint_velocity(1.5) == pytest.approx(0.2)

    joint_command = build_joint_command(
        bindings.joint_names,
        bindings.joint_group_name,
        motion.current_joint_velocity(1.5),
    )
    assert joint_command.HasField("joint_cmd")
    assert list(joint_command.joint_cmd.name) == ["finger_left", "finger_right"]
    assert list(joint_command.joint_cmd.data) == pytest.approx([0.2, 0.2])
    assert joint_command.joint_cmd.mode == core_pb2.JointCommand.ControlMode.VELOCITY
    assert joint_command.joint_cmd.group.jmg_name == "gripper"

    assert update_motion_from_key(
        motion,
        " ",
        bindings,
        linear_step=0.02,
        angular_step=0.3,
        joint_step=0.2,
        hold_until=3.0,
    )
    assert motion.current_twist(2.5) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert motion.current_joint_velocity(2.5) == 0.0
