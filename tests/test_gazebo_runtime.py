"""Tests for the Gazebo runtime manifest boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosim.core.csd import CsdGazeboRuntimeContract, CsdRealizationManifest
from robosim.gazebo_runtime import _manifest, parse_args


def test_runtime_manifest_round_trip_and_cli(tmp_path: Path) -> None:
    runtime = CsdGazeboRuntimeContract(
        world_file="world.sdf",
        robot_control_urdf="robot_control.urdf",
        controllers_file="controllers.yaml",
        namespace="/robosim/example",
        joint_state_topic="/robosim/example/joint_states",
        trajectory_action="/robosim/example/joint_trajectory_controller/follow_joint_trajectory",
        camera_topics={"Camera": "/robosim/example/cameras/Camera/image_raw"},
    )
    manifest = CsdRealizationManifest(
        manifest_id="manifest_gazebo_example",
        csd_id="example",
        backend="gazebo",
        cache_key="cache",
        root_path=str(tmp_path),
        entry_file="world.sdf",
        generated_files=("world.sdf",),
        preview_files=(),
        gazebo_runtime=runtime,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_json_dict()), encoding="utf-8")

    assert _manifest(path).gazebo_runtime == runtime
    assert parse_args(["--csd-manifest", str(path), "--no-headless"]).headless is False


def test_runtime_rejects_non_runtime_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            CsdRealizationManifest(
                manifest_id="manifest_gazebo_example",
                csd_id="example",
                backend="gazebo",
                cache_key="cache",
                root_path=str(tmp_path),
                entry_file="world.sdf",
                generated_files=("world.sdf",),
                preview_files=(),
            ).to_json_dict()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime metadata"):
        _manifest(path)

