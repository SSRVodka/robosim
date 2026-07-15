"""Device-neutral target selection for interactive servo control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2

Vector3 = tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)


class TeleopEvent(Enum):
    NEXT_TWIST_TARGET = "next_twist_target"
    NEXT_JOINT_TARGET = "next_joint_target"
    SAVE_EPISODE = "save_episode"
    RETRY_EPISODE = "retry_episode"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class TeleopMotion:
    linear: Vector3 = ZERO_VECTOR
    angular: Vector3 = ZERO_VECTOR
    joint_velocity: float = 0.0


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    motion: TeleopMotion
    events: tuple[TeleopEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class TwistTarget:
    group_name: str
    end_effector: core_pb2.EESpec
    joint_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointTarget:
    group_name: str
    joint_names: tuple[str, ...]


@dataclass(slots=True)
class TargetCatalog:
    twist_targets: tuple[TwistTarget, ...]
    joint_targets: tuple[JointTarget, ...]
    twist_index: int = 0
    joint_index: int = 0

    @property
    def active_twist(self) -> TwistTarget | None:
        return self.twist_targets[self.twist_index] if self.twist_targets else None

    @property
    def active_joint(self) -> JointTarget | None:
        return self.joint_targets[self.joint_index] if self.joint_targets else None

    def cycle_twist(self) -> core_pb2.ServoCommand | None:
        target = self.active_twist
        if target is None or len(self.twist_targets) < 2:
            return None
        command = _zero_twist_command(target)
        self.twist_index = (self.twist_index + 1) % len(self.twist_targets)
        return command

    def cycle_joint(self) -> core_pb2.ServoCommand | None:
        target = self.active_joint
        if target is None or len(self.joint_targets) < 2:
            return None
        command = _zero_joint_command(target)
        self.joint_index = (self.joint_index + 1) % len(self.joint_targets)
        return command


def build_target_catalog(
    spec: core_pb2.RobotSpecification,
    *,
    twist_targets: list[str] | None = None,
    joint_targets: list[str] | None = None,
) -> TargetCatalog:
    groups = {group.name: group for group in spec.joint_model_groups}
    twists = _resolve_twist_targets(groups, twist_targets or [])
    joints = _resolve_joint_targets(groups, joint_targets or [], twists)
    if not twists and not joints:
        raise ValueError("robot specification has no servo targets")
    return TargetCatalog(tuple(twists), tuple(joints))


def _resolve_twist_targets(
    groups: dict[str, core_pb2.JointModelGroupSpec], requested: list[str]
) -> list[TwistTarget]:
    if not requested:
        return [
            TwistTarget(group.name, ee, tuple(group.joint_names))
            for group in groups.values()
            for ee in group.end_effectors
        ]

    targets: list[TwistTarget] = []
    for value in requested:
        group_name, separator, ee_name = value.partition(":")
        group = groups.get(group_name)
        if group is None:
            raise ValueError(f"unknown twist group '{group_name}'")
        if separator:
            ee = next((entry for entry in group.end_effectors if entry.name == ee_name), None)
            if ee is None:
                raise ValueError(f"unknown end effector '{ee_name}' in group '{group_name}'")
        else:
            ee = group.end_effectors[0] if group.end_effectors else None
            if ee is None:
                raise ValueError(f"group '{group_name}' has no end effector")
        targets.append(TwistTarget(group_name, ee, tuple(group.joint_names)))
    return targets


def _resolve_joint_targets(
    groups: dict[str, core_pb2.JointModelGroupSpec],
    requested: list[str],
    twists: list[TwistTarget],
) -> list[JointTarget]:
    if requested:
        candidates = [groups.get(name) for name in requested]
        if any(group is None for group in candidates):
            unknown = next(name for name in requested if name not in groups)
            raise ValueError(f"unknown joint group '{unknown}'")
    else:
        twist_group_names = {target.group_name for target in twists}
        candidates = [
            group
            for group in groups.values()
            if group.joint_names and group.name not in twist_group_names
        ]
        candidates.sort(key=lambda group: (len(group.joint_names), group.name))

    targets: list[JointTarget] = []
    for group in candidates:
        assert group is not None
        if not group.joint_names:
            raise ValueError(f"joint group '{group.name}' has no joints")
        targets.append(JointTarget(group.name, tuple(group.joint_names)))
    return targets


def _zero_twist_command(target: TwistTarget) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        twist_cmd=core_pb2.TwistCommand(
            twist=common_pb2.TwistStamped(twist=common_pb2.Twist()),
            target_ee=target.end_effector,
        )
    )


def _zero_joint_command(target: JointTarget) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        joint_cmd=core_pb2.JointCommand(
            name=target.joint_names,
            data=[0.0] * len(target.joint_names),
            mode=core_pb2.JointCommand.ControlMode.VELOCITY,
            group=core_pb2.JointModelGroupRequest(jmg_name=target.group_name),
        )
    )
