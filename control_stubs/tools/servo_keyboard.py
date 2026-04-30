"""Keyboard client for RobotCoreService.ServoControlStream."""

from __future__ import annotations

import argparse
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Iterator
from dataclasses import dataclass

import grpc

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient

Vector3 = tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
ZERO_TWIST: tuple[Vector3, Vector3] = (ZERO_VECTOR, ZERO_VECTOR)

TWIST_KEY_BINDINGS: dict[str, tuple[Vector3, Vector3]] = {
    "w": ((1.0, 0.0, 0.0), ZERO_VECTOR),
    "s": ((-1.0, 0.0, 0.0), ZERO_VECTOR),
    "a": ((0.0, 1.0, 0.0), ZERO_VECTOR),
    "d": ((0.0, -1.0, 0.0), ZERO_VECTOR),
    "r": ((0.0, 0.0, 1.0), ZERO_VECTOR),
    "f": ((0.0, 0.0, -1.0), ZERO_VECTOR),
    "u": (ZERO_VECTOR, (1.0, 0.0, 0.0)),
    "o": (ZERO_VECTOR, (-1.0, 0.0, 0.0)),
    "i": (ZERO_VECTOR, (0.0, 1.0, 0.0)),
    "k": (ZERO_VECTOR, (0.0, -1.0, 0.0)),
    "j": (ZERO_VECTOR, (0.0, 0.0, 1.0)),
    "l": (ZERO_VECTOR, (0.0, 0.0, -1.0)),
    "\x1b[A": ((1.0, 0.0, 0.0), ZERO_VECTOR),
    "\x1b[B": ((-1.0, 0.0, 0.0), ZERO_VECTOR),
    "\x1b[D": ((0.0, 1.0, 0.0), ZERO_VECTOR),
    "\x1b[C": ((0.0, -1.0, 0.0), ZERO_VECTOR),
}
JOINT_KEY_BINDINGS = {"[": -1.0, "]": 1.0}


@dataclass(frozen=True, slots=True)
class ServoBindings:
    twist_group_name: str | None
    target_ee: core_pb2.EESpec | None
    joint_group_name: str | None
    joint_names: tuple[str, ...]
    summary_names: tuple[str, ...]


@dataclass(slots=True)
class MotionState:
    linear: Vector3 = ZERO_VECTOR
    angular: Vector3 = ZERO_VECTOR
    twist_expires_at: float = 0.0
    joint_velocity: float = 0.0
    joint_expires_at: float = 0.0

    def stop(self) -> None:
        self.linear = ZERO_VECTOR
        self.angular = ZERO_VECTOR
        self.twist_expires_at = 0.0
        self.joint_velocity = 0.0
        self.joint_expires_at = 0.0

    def current_twist(self, now: float) -> tuple[Vector3, Vector3]:
        if now < self.twist_expires_at:
            return self.linear, self.angular
        return ZERO_TWIST

    def current_joint_velocity(self, now: float) -> float:
        if now < self.joint_expires_at:
            return self.joint_velocity
        return 0.0


class CommandStream:
    def __init__(self) -> None:
        self._queue: queue.Queue[core_pb2.ServoCommand | None] = queue.Queue()

    def send(self, command: core_pb2.ServoCommand) -> None:
        self._queue.put(command)

    def close(self) -> None:
        self._queue.put(None)

    def __iter__(self) -> CommandStream:
        return self

    def __next__(self) -> core_pb2.ServoCommand:
        command = self._queue.get()
        if command is None:
            raise StopIteration
        return command


def _scale_vector(vector: Vector3, scale: float) -> Vector3:
    return tuple(value * scale for value in vector)  # type: ignore[return-value]


def _find_group(
    spec: core_pb2.RobotSpecification, name: str
) -> core_pb2.JointModelGroupSpec | None:
    for group in spec.joint_model_groups:
        if group.name == name:
            return group
    return None


def _find_group_for_ee(
    spec: core_pb2.RobotSpecification, ee_name: str
) -> core_pb2.JointModelGroupSpec | None:
    for group in spec.joint_model_groups:
        if any(ee.name == ee_name for ee in group.end_effectors):
            return group
    return None


def select_servo_bindings(
    spec: core_pb2.RobotSpecification,
    twist_group_name: str | None = None,
    ee_name: str | None = None,
    joint_group_name: str | None = None,
) -> ServoBindings:
    twist_group: core_pb2.JointModelGroupSpec | None = None
    if twist_group_name:
        twist_group = _find_group(spec, twist_group_name)
        if twist_group is None:
            raise ValueError(f"Unknown twist group: {twist_group_name}")
    elif ee_name:
        twist_group = _find_group_for_ee(spec, ee_name)
        if twist_group is None:
            raise ValueError(f"Unable to find end effector: {ee_name}")
    else:
        twist_group = next(
            (group for group in spec.joint_model_groups if group.end_effectors),
            None,
        )

    target_ee: core_pb2.EESpec | None = None
    if twist_group is not None:
        if ee_name:
            target_ee = next((ee for ee in twist_group.end_effectors if ee.name == ee_name), None)
            if target_ee is None:
                raise ValueError(
                    f"End effector '{ee_name}' does not belong to group '{twist_group.name}'"
                )
        elif twist_group.end_effectors:
            target_ee = twist_group.end_effectors[0]
        elif twist_group_name:
            raise ValueError(f"Group '{twist_group.name}' has no end effector")

    joint_group: core_pb2.JointModelGroupSpec | None = None
    if joint_group_name:
        joint_group = _find_group(spec, joint_group_name)
        if joint_group is None:
            raise ValueError(f"Unknown joint group: {joint_group_name}")
        if not joint_group.joint_names:
            raise ValueError(f"Joint group '{joint_group_name}' has no joints")
    else:
        candidates = [
            group
            for group in spec.joint_model_groups
            if group.joint_names and group.name != (twist_group.name if twist_group else None)
        ]
        if candidates:
            ee_candidates = [group for group in candidates if group.end_effectors]
            joint_group = min(
                ee_candidates or candidates,
                key=lambda group: (len(group.joint_names), group.name),
            )

    if target_ee is None and joint_group is None:
        raise ValueError(
            "Robot specification has neither end effector nor joint group for servoing"
        )

    summary_names = tuple(joint_group.joint_names) if joint_group else ()
    if not summary_names and twist_group is not None:
        summary_names = tuple(twist_group.joint_names)

    return ServoBindings(
        twist_group_name=twist_group.name if twist_group else None,
        target_ee=target_ee,
        joint_group_name=joint_group.name if joint_group else None,
        joint_names=tuple(joint_group.joint_names) if joint_group else (),
        summary_names=summary_names,
    )


def build_twist_command(
    target_ee: core_pb2.EESpec, linear: Vector3, angular: Vector3
) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        twist_cmd=core_pb2.TwistCommand(
            twist=common_pb2.TwistStamped(
                twist=common_pb2.Twist(
                    linear=common_pb2.Point(x=linear[0], y=linear[1], z=linear[2]),
                    angular=common_pb2.Point(x=angular[0], y=angular[1], z=angular[2]),
                )
            ),
            target_ee=target_ee,
        )
    )


def build_joint_command(
    joint_names: tuple[str, ...], joint_group_name: str, velocity: float
) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        joint_cmd=core_pb2.JointCommand(
            name=list(joint_names),
            data=[velocity] * len(joint_names),
            mode=core_pb2.JointCommand.ControlMode.VELOCITY,
            group=core_pb2.JointModelGroupRequest(jmg_name=joint_group_name),
        )
    )


def update_motion_from_key(
    motion: MotionState,
    key: str,
    bindings: ServoBindings,
    *,
    linear_step: float,
    angular_step: float,
    joint_step: float,
    hold_until: float,
) -> bool:
    if key == " ":
        motion.stop()
        return True
    if key in TWIST_KEY_BINDINGS and bindings.target_ee is not None:
        linear_axis, angular_axis = TWIST_KEY_BINDINGS[key]
        motion.linear = _scale_vector(linear_axis, linear_step)
        motion.angular = _scale_vector(angular_axis, angular_step)
        motion.twist_expires_at = hold_until
        return True
    if key in JOINT_KEY_BINDINGS and bindings.joint_names:
        motion.joint_velocity = JOINT_KEY_BINDINGS[key] * joint_step
        motion.joint_expires_at = hold_until
        return True
    return False


def format_robot_spec(spec: core_pb2.RobotSpecification) -> str:
    lines = [f"robot: {spec.robot_name}"]
    for group in spec.joint_model_groups:
        ee_names = ", ".join(ee.name for ee in group.end_effectors) or "-"
        lines.append(
            f"  {group.name}: joints={len(group.joint_names)} ee={ee_names}"
        )
    return "\n".join(lines)


def summarize_joint_state(state: common_pb2.JointState, names: tuple[str, ...]) -> str:
    positions = dict(zip(state.name, state.position, strict=False))
    selected = names[:6] if names else tuple(state.name[:6])
    if not selected:
        return f"state: joints={len(state.name)}"
    summary = ", ".join(
        f"{name}={positions.get(name, float('nan')):.3f}" for name in selected
    )
    return f"state: {summary}"


def _print_controls(bindings: ServoBindings, args: argparse.Namespace) -> None:
    ee_name = bindings.target_ee.name if bindings.target_ee is not None else "-"
    print("ServoControlStream keyboard client")
    print(
        "twist target:",
        f"group={bindings.twist_group_name or '-'} ee={ee_name} "
        f"linear_step={args.linear_step:.4f} angular_step={args.angular_step:.4f}",
    )
    if bindings.joint_names:
        print(
            "joint target:",
            f"group={bindings.joint_group_name} joints={list(bindings.joint_names)} "
            f"joint_step={args.joint_step:.4f}",
        )
    else:
        print("joint target: disabled")
    print("keys: w/s x, a/d y, r/f z, u/o roll, i/k pitch, j/l yaw, arrows for planar twist")
    print("keys: [/ ] joint -, +, space stop, q quit")


def _read_key(fd: int, timeout: float) -> str | None:
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    data = os.read(fd, 1)
    if not data:
        return None
    key = data.decode(errors="ignore")
    if key != "\x1b":
        return key
    ready, _, _ = select.select([fd], [], [], 0.01)
    if not ready:
        return key
    return key + os.read(fd, 2).decode(errors="ignore")


def _consume_states(
    responses: Iterator[common_pb2.JointState],
    summary_names: tuple[str, ...],
    stop_event: threading.Event,
) -> None:
    try:
        for state in responses:
            sys.stdout.write(f"\r\033[K{summarize_joint_state(state, summary_names)}")
            sys.stdout.flush()
    except grpc.RpcError as exc:
        sys.stdout.write("\n")
        print(f"stream closed: {exc.code().name}: {exc.details()}", file=sys.stderr)
    finally:
        stop_event.set()


def _emit_commands(
    stream: CommandStream,
    bindings: ServoBindings,
    motion: MotionState,
    now: float,
    last_twist: tuple[Vector3, Vector3],
    last_joint_velocity: float,
) -> tuple[tuple[Vector3, Vector3], float]:
    current_twist = motion.current_twist(now)
    if bindings.target_ee is not None and current_twist != last_twist:
        stream.send(build_twist_command(bindings.target_ee, *current_twist))
        last_twist = current_twist

    current_joint_velocity = motion.current_joint_velocity(now)
    if bindings.joint_names and current_joint_velocity != last_joint_velocity:
        assert bindings.joint_group_name is not None
        stream.send(
            build_joint_command(
                bindings.joint_names,
                bindings.joint_group_name,
                current_joint_velocity,
            )
        )
        last_joint_velocity = current_joint_velocity

    return last_twist, last_joint_velocity


def run(args: argparse.Namespace) -> int:
    if args.rate <= 0.0:
        raise ValueError("--rate must be > 0")
    if args.hold_time <= 0.0:
        raise ValueError("--hold-time must be > 0")
    client = RobosimClient(args.host, args.port)
    try:
        spec = client.robot_core.get_robot_spec()
        if args.list:
            print(format_robot_spec(spec))
            return 0
        bindings = select_servo_bindings(spec, args.jmg, args.ee, args.joint_group)
        stream = CommandStream()
        responses = client.robot_core.servo_control_stream(stream)
        stop_event = threading.Event()
        response_thread = threading.Thread(
            target=_consume_states,
            args=(responses, bindings.summary_names, stop_event),
            daemon=True,
        )
        response_thread.start()
        _print_controls(bindings, args)

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        motion = MotionState()
        last_twist = ZERO_TWIST
        last_joint_velocity = 0.0
        period = 1.0 / args.rate

        try:
            tty.setraw(fd)
            while not stop_event.is_set():
                key = _read_key(fd, period)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "q":
                    break
                if key is not None:
                    update_motion_from_key(
                        motion,
                        key,
                        bindings,
                        linear_step=args.linear_step,
                        angular_step=args.angular_step,
                        joint_step=args.joint_step,
                        hold_until=time.monotonic() + args.hold_time,
                    )
                last_twist, last_joint_velocity = _emit_commands(
                    stream,
                    bindings,
                    motion,
                    time.monotonic(),
                    last_twist,
                    last_joint_velocity,
                )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
            motion.stop()
            last_twist, last_joint_velocity = _emit_commands(
                stream,
                bindings,
                motion,
                time.monotonic(),
                last_twist,
                last_joint_velocity,
            )
            stream.close()
            response_thread.join(timeout=1.0)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return 0
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--jmg",
        help="Joint model group used for twist servo target selection",
    )
    parser.add_argument("--ee", help="End effector name used for twist servo target selection")
    parser.add_argument(
        "--joint-group",
        help="Joint model group used for direct joint velocity servo",
    )
    parser.add_argument("--linear-step", type=float, default=0.02)
    parser.add_argument("--angular-step", type=float, default=0.3)
    parser.add_argument("--joint-step", type=float, default=0.2)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--hold-time", type=float, default=0.15)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print robot groups/end effectors and exit",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
