"""Tests for the server-owned Gazebo runtime."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from robosim.backends.gazebo.runtime import GazeboRuntime
from robosim.core.csd import CsdGazeboRuntimeContract, CsdRealizationManifest


def _manifest(tmp_path: Path) -> Path:
    runtime = CsdGazeboRuntimeContract(
        world_file="world.sdf",
        robot_control_urdf="robot_control.urdf",
        controllers_file="controllers.yaml",
        namespace="/robosim/example",
        joint_state_topic="/robosim/example/joint_states",
        trajectory_action="/robosim/example/arm/follow_joint_trajectory",
        camera_topics={},
        robot_semantics_file="runtime/robot_semantics.json",
        trajectory_actions={"arm": "/robosim/example/arm/follow_joint_trajectory"},
    )
    manifest = CsdRealizationManifest(
        manifest_id="manifest_gazebo_example",
        csd_id="example",
        backend="gazebo",
        cache_key="cache",
        root_path="/stale/package/path",
        entry_file="world.sdf",
        generated_files=("world.sdf",),
        preview_files=(),
        gazebo_runtime=runtime,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_json_dict()), encoding="utf-8")
    for relative in ("world.sdf", "robot_control.urdf", "controllers.yaml"):
        (tmp_path / relative).touch()
    semantics = tmp_path / "runtime" / "robot_semantics.json"
    semantics.parent.mkdir()
    semantics.write_text(
        json.dumps(
            {
                "controller_names": {"arm": "arm_controller"},
                "controller_groups": {"arm": ["joint1"]},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("headless, expected_processes", ((True, 2), (False, 3)))
def test_runtime_headless_controls_gzclient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    headless: bool,
    expected_processes: int,
) -> None:
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, command: tuple[str, ...], **_: object) -> None:
            commands.append(command)
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.running = False

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0)
    )
    runtime = GazeboRuntime(_manifest(tmp_path), headless=headless)

    runtime.start()
    runtime.stop()

    assert len(commands) == expected_processes
    assert any(Path(command[0]).name == "gzserver" for command in commands)
    assert any(Path(command[0]).name == "gzclient" for command in commands) is (not headless)


def test_runtime_rejects_non_gazebo_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            CsdRealizationManifest(
                manifest_id="manifest_mujoco_example",
                csd_id="example",
                backend="mujoco",
                cache_key="cache",
                root_path=str(tmp_path),
                entry_file="scene.xml",
                generated_files=("scene.xml",),
                preview_files=(),
            ).to_json_dict()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gazebo realization"):
        GazeboRuntime(path, headless=True)
