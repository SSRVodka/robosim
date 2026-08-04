from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control_stubs.tools.traj_synthesis import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PANDA_VELOCITY_LIMIT,
    SCHEMA_BLOCKER,
    InstructionStream,
    MinkIK,
    SynthesisConfig,
    SynthesisError,
    synthesize,
    validate,
)

PANDA_SCENE = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim"
    / "mujoco"
    / "assets"
    / "robots"
    / "franka_panda"
    / "scene.xml"
)
OBJECTS = {
    "cup": np.array([0.6, -0.2, 0.151]),
    "container": np.array([0.4, 0.3, 0.15]),
}


@pytest.fixture(scope="module")
def ik() -> MinkIK:
    return MinkIK(PANDA_SCENE, SynthesisConfig())


@pytest.fixture(scope="module")
def stream(ik: MinkIK) -> InstructionStream:
    return synthesize(ik, OBJECTS, seed=0)


def test_synthesized_stream_converges_and_passes_gate(
    ik: MinkIK, stream: InstructionStream
) -> None:
    validate(ik, stream, cup_pose=OBJECTS["cup"])

    times = np.array([sample.t for sample in stream.samples])
    assert len(stream.samples) > 100
    assert np.allclose(np.diff(times), 1.0 / 30.0)

    grippers = [sample.gripper for sample in stream.samples]
    close_index = grippers.index(GRIPPER_CLOSED)
    assert 0 < close_index < len(grippers) - 1
    assert grippers[-1] == GRIPPER_OPEN  # 以释放收尾

    worst = stream.meta["ik_worst_residual"]
    assert worst["pos"] < SynthesisConfig().ik_pos_tol


def test_gate_rejects_velocity_violation(ik: MinkIK) -> None:
    fast = SynthesisConfig(cartesian_vmax=3.0, cartesian_amax=10.0, cartesian_jmax=50.0)
    fast_stream = synthesize(ik, OBJECTS, config=fast, seed=0)
    with pytest.raises(SynthesisError) as excinfo:
        validate(ik, fast_stream, config=fast, cup_pose=OBJECTS["cup"])
    assert excinfo.value.payload["stage"] == "joint_velocity"


def test_gate_rejects_collision_on_obstructed_descent(
    ik: MinkIK, stream: InstructionStream
) -> None:
    with pytest.raises(SynthesisError) as excinfo:
        validate(ik, stream, cup_pose=OBJECTS["cup"] + np.array([0.0, 0.0, 0.06]))
    payload = excinfo.value.payload
    assert payload["stage"] == "collision"
    assert payload["schema"] == SCHEMA_BLOCKER


def test_unreachable_target_raises_typed_error(ik: MinkIK) -> None:
    unreachable = {"cup": np.array([1.4, 0.0, 0.151]), "container": OBJECTS["container"]}
    with pytest.raises(SynthesisError) as excinfo:
        synthesize(ik, unreachable, seed=0)
    assert excinfo.value.payload["stage"] == "ik_convergence"


def test_instruction_stream_roundtrip(
    stream: InstructionStream, tmp_path: Path
) -> None:
    path = tmp_path / "stream.jsonl"
    stream.write(path)
    loaded = InstructionStream.read(path)
    assert len(loaded.samples) == len(stream.samples)
    assert loaded.meta["fps"] == 30
    np.testing.assert_allclose(
        loaded.samples[-1].q, stream.samples[-1].q, atol=1e-6
    )


def test_ruckig_segments_compute_locally(
    ik: MinkIK, capfd: pytest.CaptureFixture[str]
) -> None:
    # 红线回归：社区版 ruckig 在设置 intermediate_positions 时会静默调云 API
    # 并打印 "[ruckig] calculate trajectory via cloud API."。本模块只允许
    # state-to-state 单段;任何云调用日志都视为违规。
    synthesize(ik, OBJECTS, seed=1)
    captured = capfd.readouterr()
    assert "cloud" not in captured.out.lower()
    assert "cloud" not in captured.err.lower()


def test_velocity_stays_within_official_limits(stream: InstructionStream) -> None:
    q_matrix = np.stack([sample.q for sample in stream.samples])
    step_velocity = np.abs(np.diff(q_matrix, axis=0)) * 30.0
    assert (step_velocity <= PANDA_VELOCITY_LIMIT).all()
