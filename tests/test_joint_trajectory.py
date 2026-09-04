"""Tests for server-side joint trajectory execution on the MuJoCo backend."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from robosim.backends.mujoco.backend import MuJoCoBackend

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)
ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def _arm_positions(backend: MuJoCoBackend) -> dict[str, float]:
    state = backend.get_robot_state()
    return dict(zip(state.name, state.position, strict=True))


def test_trajectory_executes_by_sim_time(backend: MuJoCoBackend) -> None:
    home = [_arm_positions(backend)[name] for name in ARM_JOINTS]
    target = list(home)
    target[0] += 0.3
    steps = 20
    points = [
        (
            0.02 * (index + 1),
            [h + (t - h) * (index + 1) / steps for h, t in zip(home, target, strict=True)],
        )
        for index in range(steps)
    ]
    # Hold the final target for a while so the PD controller settles.
    points.append((points[-1][0] + 0.5, target))

    start_sim_time = backend.get_robot_state().header.timestamp
    backend.execute_joint_trajectory(ARM_JOINTS, points, "panda_arm")
    elapsed_sim = backend.get_robot_state().header.timestamp - start_sim_time

    assert elapsed_sim >= points[-1][0]
    final = _arm_positions(backend)
    assert abs(final[ARM_JOINTS[0]] - target[0]) < 5e-3


def test_trajectory_rejects_bad_input(backend: MuJoCoBackend) -> None:
    with pytest.raises(ValueError):
        backend.execute_joint_trajectory(ARM_JOINTS, [], "panda_arm")
    with pytest.raises(ValueError):
        backend.execute_joint_trajectory(
            ARM_JOINTS, [(0.1, [0.0] * 7), (0.05, [0.0] * 7)], "panda_arm"
        )
    with pytest.raises(ValueError):
        backend.execute_joint_trajectory(ARM_JOINTS, [(0.1, [0.0] * 3)], "panda_arm")
    with pytest.raises(ValueError):
        backend.execute_joint_trajectory(["nonexistent"], [(0.1, [0.0])], "panda_arm")


def test_trajectory_respects_pause(backend: MuJoCoBackend) -> None:
    home = [_arm_positions(backend)[name] for name in ARM_JOINTS]
    done = threading.Event()

    def run() -> None:
        backend.execute_joint_trajectory(ARM_JOINTS, [(0.3, home)], "panda_arm")
        done.set()

    backend._paused = True
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert not done.wait(0.5)  # paused simulation never advances the trajectory
    backend._paused = False
    assert done.wait(5.0)
    worker.join()


def test_concurrent_trajectories_are_rejected(backend: MuJoCoBackend) -> None:
    home = [_arm_positions(backend)[name] for name in ARM_JOINTS]
    first_started = threading.Event()

    def run() -> None:
        first_started.set()
        backend.execute_joint_trajectory(ARM_JOINTS, [(1.0, home)], "panda_arm")

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    first_started.wait(1.0)
    time.sleep(0.1)
    with pytest.raises(RuntimeError):
        backend.execute_joint_trajectory(ARM_JOINTS, [(0.1, home)], "panda_arm")
    worker.join()


def test_trajectory_targets_visible_in_command_state(backend: MuJoCoBackend) -> None:
    home = np.array([_arm_positions(backend)[name] for name in ARM_JOINTS])
    target = home.copy()
    target[1] -= 0.2
    backend.execute_joint_trajectory(
        ARM_JOINTS, [(0.05, list(home)), (0.4, list(target))], "panda_arm"
    )
    command = backend.get_joint_command_state()
    command_by_name = dict(zip(command.name, command.position, strict=True))
    assert abs(command_by_name[ARM_JOINTS[1]] - target[1]) < 1e-9
