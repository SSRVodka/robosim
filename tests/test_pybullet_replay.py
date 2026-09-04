from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control_stubs.tools import replay_stream_pybullet
from control_stubs.tools.traj_synthesis import (
    MinkIK,
    SynthesisConfig,
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
PYBULLET_SCENE = Path(__file__).resolve().parent / "fixtures" / "pybullet_cup_scene"


@pytest.fixture(scope="module")
def stream_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ik = MinkIK(PANDA_SCENE, SynthesisConfig())
    objects = {
        "cup": np.array([0.62, -0.22, 0.151]),
        "container": np.array([0.4, 0.3, 0.15]),
    }
    stream = synthesize(ik, objects, seed=7)
    validate(ik, stream, cup_pose=objects["cup"])
    path = tmp_path_factory.mktemp("streams") / "seed00007.jsonl"
    stream.write(path)
    return path


def test_stream_replays_on_pybullet_with_local_success_judgment(
    stream_path: Path,
) -> None:
    result = replay_stream_pybullet.replay(
        stream_path,
        PYBULLET_SCENE / "scene.py",
        PYBULLET_SCENE / "scene_meta.json",
    )
    assert result["backend"] == "pybullet"
    assert result["frames"] > 100
    # 机械臂关节轨迹跨后端可对齐（任务级语义由谓词独立判定）
    assert result["max_arm_tracking_error_rad"] < 0.05
    assert isinstance(result["success"], bool)
    assert result["success"], f"pick-place failed on pybullet: {result}"
