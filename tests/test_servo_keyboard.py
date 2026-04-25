from __future__ import annotations

import pytest

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.servo_keyboard import (
    MotionState,
    build_joint_command,
    build_twist_command,
    select_servo_bindings,
    update_motion_from_key,
)


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


def test_select_servo_bindings_rejects_invalid_explicit_group() -> None:
    with pytest.raises(ValueError, match="has no end effector"):
        select_servo_bindings(_make_spec(), twist_group_name="gripper")


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
