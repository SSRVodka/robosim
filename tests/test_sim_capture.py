"""Tests for simulation-time-aligned capture on the MuJoCo backend."""

from __future__ import annotations

import itertools
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from robosim.backends.mujoco.backend import SIM_CAPTURE_BUFFER_COUNT, MuJoCoBackend

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)
CAPTURE_FPS = 50.0


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def test_capture_samples_align_with_sim_time(backend: MuJoCoBackend) -> None:
    backend.start_sim_capture(CAPTURE_FPS)
    snapshots = []
    while len(snapshots) < 20:
        snapshot = backend.next_sim_capture(["panda_arm"], [])
        assert snapshot is not None
        snapshots.append(snapshot)
    backend.stop_sim_capture()

    timestamps = [snap.robot_state.header.timestamp for snap in snapshots]
    interval = 1.0 / CAPTURE_FPS
    timestep = backend._model.opt.timestep
    # Frames land on the first physics-step boundary at/after each absolute
    # sample time, so per-pair jitter is at most one timestep and never drifts.
    for earlier, later in itertools.pairwise(timestamps):
        assert interval - timestep - 1e-9 <= later - earlier <= interval + timestep + 1e-9
    total_span = timestamps[-1] - timestamps[0]
    assert abs(total_span - (len(timestamps) - 1) * interval) <= timestep + 1e-9
    assert all("panda_arm" in snap.end_effector_states for snap in snapshots)
    assert len(snapshots[0].robot_state.position) == 9


def test_capture_backpressure_pauses_simulation(backend: MuJoCoBackend) -> None:
    backend.start_sim_capture(CAPTURE_FPS)
    start_time = backend.get_robot_state().header.timestamp
    time.sleep(0.6)
    stalled_time = backend.get_robot_state().header.timestamp
    # Without a consumer, sim time may only advance until the buffer pool is
    # exhausted (one interval per pooled buffer, plus the pending step).
    max_advance = (SIM_CAPTURE_BUFFER_COUNT + 1) / CAPTURE_FPS + 0.01
    assert stalled_time - start_time <= max_advance

    drained = 0
    while backend.next_sim_capture([], []) is not None and drained < SIM_CAPTURE_BUFFER_COUNT:
        drained += 1
    backend.stop_sim_capture()
    assert drained == SIM_CAPTURE_BUFFER_COUNT


def test_capture_stop_unblocks_consumer(backend: MuJoCoBackend) -> None:
    backend.start_sim_capture(CAPTURE_FPS)
    assert backend.next_sim_capture([], []) is not None
    backend.stop_sim_capture()
    assert backend.next_sim_capture([], []) is None
    # Simulation resumes normal stepping after capture stops.
    resume_time = backend.get_robot_state().header.timestamp
    time.sleep(0.2)
    assert backend.get_robot_state().header.timestamp > resume_time
