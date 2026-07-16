"""Device-neutral target selection for interactive servo control."""

from __future__ import annotations

import argparse
import queue
import select
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import grpc

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.client import RobosimClient

Vector3 = tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)

if __name__ == "__main__":
    sys.modules["control_stubs.tools.teleop"] = sys.modules[__name__]


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

    @property
    def record_group_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for target in (*self.twist_targets, *self.joint_targets):
            if target.group_name not in names:
                names.append(target.group_name)
        return tuple(names)


class ServoSession:
    """Apply device-neutral motion snapshots to the active servo targets."""

    def __init__(
        self,
        targets: TargetCatalog,
        send: Callable[[core_pb2.ServoCommand], None],
    ) -> None:
        self.targets = targets
        self._send = send
        self._last_twist: tuple[Vector3, Vector3] = (ZERO_VECTOR, ZERO_VECTOR)
        self._last_joint_velocity = 0.0

    def apply(self, snapshot: InputSnapshot) -> tuple[TeleopEvent, ...]:
        remaining: list[TeleopEvent] = []
        for event in snapshot.events:
            if event is TeleopEvent.NEXT_TWIST_TARGET:
                if command := self.targets.cycle_twist():
                    self._send(command)
                    self._last_twist = (ZERO_VECTOR, ZERO_VECTOR)
            elif event is TeleopEvent.NEXT_JOINT_TARGET:
                if command := self.targets.cycle_joint():
                    self._send(command)
                    self._last_joint_velocity = 0.0
            else:
                remaining.append(event)

        motion = snapshot.motion
        if any(
            event
            in (TeleopEvent.SAVE_EPISODE, TeleopEvent.RETRY_EPISODE, TeleopEvent.STOP)
            for event in remaining
        ):
            motion = TeleopMotion()
        target = self.targets.active_twist
        twist = (motion.linear, motion.angular)
        if target is not None and twist != self._last_twist:
            self._send(_twist_command(target, motion.linear, motion.angular))
            self._last_twist = twist

        joint_target = self.targets.active_joint
        if (
            joint_target is not None
            and motion.joint_velocity != self._last_joint_velocity
        ):
            self._send(_joint_command(joint_target, motion.joint_velocity))
            self._last_joint_velocity = motion.joint_velocity
        return tuple(remaining)

    def stop(self) -> None:
        if target := self.targets.active_twist:
            self._send(_zero_twist_command(target))
        if target := self.targets.active_joint:
            self._send(_zero_joint_command(target))
        self._last_twist = (ZERO_VECTOR, ZERO_VECTOR)
        self._last_joint_velocity = 0.0


@dataclass(frozen=True, slots=True)
class RecordingConfig:
    repo_name: str
    task_text: str
    fps: int
    jmg_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    reset_between_episodes: bool = False


class EpisodeController:
    """Sequence recorder and optional reset RPCs between teleop episodes."""

    def __init__(self, client: RobosimClient, config: RecordingConfig) -> None:
        self._client = client
        self._config = config
        self._active = False

    def start(self) -> None:
        job = self._client.robot_data.episode_start(
            repo_name=self._config.repo_name,
            task_text=self._config.task_text,
            fps=self._config.fps,
            jmg_included=list(self._config.jmg_names),
            sensor_name_included=list(self._config.sensor_names),
        )
        self._require_success(job.status, "start episode")
        self._active = True

    def handle(self, event: TeleopEvent) -> None:
        if event not in (TeleopEvent.SAVE_EPISODE, TeleopEvent.RETRY_EPISODE):
            return
        if not self._active:
            raise RuntimeError("recording is not active")
        if event is TeleopEvent.SAVE_EPISODE:
            status = self._client.robot_data.episode_end()
            action = "save episode"
        else:
            status = self._client.robot_data.episode_cancel()
            action = "cancel episode"
        self._require_success(status, action)
        self._active = False
        if self._config.reset_between_episodes:
            self._require_success(self._client.simulation.reset_world(), "reset world")
        self.start()

    def stop(self) -> None:
        if not self._active:
            return
        status = self._client.robot_data.episode_cancel()
        self._active = False
        self._require_success(status, "cancel episode")

    @staticmethod
    def _require_success(status: common_pb2.Status, action: str) -> None:
        if status.code != common_pb2.STATUS_SUCCESS:
            raise RuntimeError(f"failed to {action}: {status.message}")


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


def build_parser() -> argparse.ArgumentParser:
    from control_stubs.tools.servo_keyboard import build_parser as build_keyboard_parser

    parser = build_keyboard_parser()
    parser.description = "Interactive keyboard/Joy-Con teleoperation and data collection"
    parser.add_argument("--input", choices=("keyboard", "joycon"), default="keyboard")
    parser.add_argument("--input-device", help="Linux evdev path used by Joy-Con input")
    parser.add_argument("--input-profile", choices=("joycon-right",), default="joycon-right")
    parser.add_argument("--deadzone", type=float, default=0.1)
    parser.add_argument("--repo-name", help="LeRobotDataset repo name; enables recording")
    parser.add_argument("--task-text", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sensor", action="append", default=[], metavar="NAME")
    parser.add_argument("--reset-between-episodes", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.input == "keyboard":
        from control_stubs.tools.servo_keyboard import run as run_keyboard

        return run_keyboard(args)
    return _run_joycon(args)


def _run_joycon(args: argparse.Namespace) -> int:
    from control_stubs.tools.joycon import JoyConInput

    if not args.input_device:
        raise ValueError("--input-device is required for Joy-Con input")
    if args.rate <= 0.0:
        raise ValueError("--rate must be > 0")
    client = RobosimClient(args.host, args.port)
    input_device: JoyConInput | None = None
    stream = CommandStream()
    stop_event = threading.Event()
    response_thread: threading.Thread | None = None
    episode: EpisodeController | None = None
    try:
        spec = client.robot_core.get_robot_spec()
        if args.list:
            from control_stubs.tools.servo_keyboard import format_robot_spec

            print(format_robot_spec(spec))
            return 0
        input_device = JoyConInput.open(
            args.input_device,
            deadzone=args.deadzone,
            linear_speed=args.linear_step,
            angular_speed=args.angular_step,
            joint_speed=args.joint_step,
        )
        targets = build_target_catalog(
            spec,
            twist_targets=list(args.twist_target),
            joint_targets=list(args.joint_target),
        )
        session = ServoSession(targets, stream.send)
        responses = client.robot_core.servo_control_stream(stream)
        response_thread = threading.Thread(
            target=_drain_responses,
            args=(responses, stop_event),
            daemon=True,
        )
        response_thread.start()
        if args.repo_name:
            episode = EpisodeController(
                client,
                RecordingConfig(
                    repo_name=args.repo_name,
                    task_text=args.task_text,
                    fps=args.fps,
                    jmg_names=targets.record_group_names,
                    sensor_names=tuple(args.sensor),
                    reset_between_episodes=args.reset_between_episodes,
                ),
            )
            episode.start()

        print("Joy-Con teleop: A save, B retry, Plus quit, Home/stick switch targets")
        period = 1.0 / args.rate
        should_stop = False
        while not should_stop and not stop_event.is_set():
            ready, _, _ = select.select([input_device.fd], [], [], period)
            if not ready:
                continue
            for event in session.apply(input_device.read()):
                if event is TeleopEvent.STOP:
                    should_stop = True
                elif episode is not None:
                    episode.handle(event)
    finally:
        try:
            if "session" in locals():
                session.stop()
        finally:
            try:
                if episode is not None:
                    episode.stop()
            finally:
                try:
                    if input_device is not None:
                        input_device.close()
                finally:
                    stream.close()
                    if response_thread is not None:
                        response_thread.join(timeout=1.0)
                    client.close()
    return 0


def _drain_responses(
    responses: Iterator[common_pb2.JointState], stop_event: threading.Event
) -> None:
    try:
        for _ in responses:
            pass
    except grpc.RpcError as exc:
        print(f"stream closed: {exc.code().name}: {exc.details()}", file=sys.stderr)
    finally:
        stop_event.set()


def main() -> int:
    return run(build_parser().parse_args())


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
    return _twist_command(target, ZERO_VECTOR, ZERO_VECTOR)


def _zero_joint_command(target: JointTarget) -> core_pb2.ServoCommand:
    return _joint_command(target, 0.0)


def _twist_command(
    target: TwistTarget, linear: Vector3, angular: Vector3
) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        twist_cmd=core_pb2.TwistCommand(
            twist=common_pb2.TwistStamped(
                twist=common_pb2.Twist(
                    linear=common_pb2.Point(x=linear[0], y=linear[1], z=linear[2]),
                    angular=common_pb2.Point(x=angular[0], y=angular[1], z=angular[2]),
                )
            ),
            target_ee=target.end_effector,
        )
    )


def _joint_command(target: JointTarget, velocity: float) -> core_pb2.ServoCommand:
    return core_pb2.ServoCommand(
        joint_cmd=core_pb2.JointCommand(
            name=target.joint_names,
            data=[velocity] * len(target.joint_names),
            mode=core_pb2.JointCommand.ControlMode.VELOCITY,
            group=core_pb2.JointModelGroupRequest(jmg_name=target.group_name),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
