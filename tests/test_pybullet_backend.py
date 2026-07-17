"""Tests for the PyBullet backend."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator, Mapping
from pathlib import Path

import numpy as np
import pytest

from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.common_pb2 import Point, Pose, Quaternion
from control_stubs.robot_core_pb2 import ServoCommand
from control_stubs.robot_data_pb2 import RecordInfo, RecordOptions
from robosim.backends.pybullet.backend import PyBulletBackend
from robosim.core import (
    CsdRealizationManifest,
    compile_csd_to_pybullet,
)
from robosim.core.impl.recorder_lerobot import LerobotDataRecorder

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"
SHARED_OPENUSD_CSD = FIXTURE_ROOT / "openusd" / "shared_tabletop" / "csd.usda"
SEMANTIC_OPENUSD_ROOT = FIXTURE_ROOT / "openusd" / "semantic"


def _csd_fixture(name: str) -> Path:
    return SEMANTIC_OPENUSD_ROOT / name.removesuffix(".json") / "csd.usda"


@pytest.fixture
def backend() -> Generator[PyBulletBackend, None, None]:
    instance = PyBulletBackend(headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def _wait_for_condition(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _load_registry_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _fixture_mesh_half_extents(path: Path) -> tuple[float, float, float]:
    name = path.stem
    if "tray" in name:
        return (0.08, 0.055, 0.012)
    if "mug" in name:
        return (0.035, 0.035, 0.055)
    return (0.035, 0.035, 0.035)


def _write_box_mesh(path: Path) -> None:
    hx, hy, hz = _fixture_mesh_half_extents(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"v {-hx} {-hy} {-hz}",
                f"v {hx} {-hy} {-hz}",
                f"v {hx} {hy} {-hz}",
                f"v {-hx} {hy} {-hz}",
                f"v {-hx} {-hy} {hz}",
                f"v {hx} {-hy} {hz}",
                f"v {hx} {hy} {hz}",
                f"v {-hx} {hy} {hz}",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            )
        ),
        encoding="utf-8",
    )


def _write_fixture_asset_files(asset_root: Path, asset_registry: Mapping[str, object]) -> None:
    records = asset_registry.get("objects", ())
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for variant in record.get("backend_resources", ()):
            if isinstance(variant, Mapping) and variant.get("mesh_path"):
                _write_box_mesh(asset_root / str(variant["mesh_path"]))


def _image_array(image) -> np.ndarray:
    return np.frombuffer(image.data, dtype=np.uint8).reshape(
        int(image.height),
        int(image.width),
        3,
    )


def test_default_pybullet_backend_starts_and_renders(backend: PyBulletBackend) -> None:
    spec = backend.get_robot_spec()
    sensors = {entry.name: entry.type for entry in backend.list_sensors().entries}

    assert spec.robot_name == "panda"
    assert "panda_arm" in {group.name for group in spec.joint_model_groups}
    assert "joint_states" in sensors
    assert "world_camera" in sensors

    image = backend.get_sensors(["world_camera"]).images[0]

    assert image.width == 320
    assert image.height == 240
    assert float(_image_array(image).std()) > 1.0


def test_pybullet_backend_position_control_and_servo_stream(backend: PyBulletBackend) -> None:
    before = backend.get_robot_state()
    index = list(before.name).index("panda_joint1")
    target = float(before.position[index]) + 0.08

    backend.set_joint_target(
        ["panda_joint1"],
        [target],
        core_pb2.JointCommand.ControlMode.POSITION,
        "panda_arm",
    )

    assert _wait_for_condition(
        lambda: abs(float(backend.get_robot_state().position[index]) - target) < 0.03,
        timeout=2.0,
    )
    command_state = backend.get_joint_command_state()
    assert "panda_joint1" in command_state.name

    requests = iter(
        [
            ServoCommand(
                joint_cmd=core_pb2.JointCommand(
                    name=["panda_joint1"],
                    data=[target],
                    mode=core_pb2.JointCommand.ControlMode.POSITION,
                    group=core_pb2.JointModelGroupRequest(jmg_name="panda_arm"),
                )
            )
        ]
    )
    response = next(backend.servo_control_stream(requests))
    assert "panda_joint1" in response.name


def test_pybullet_backend_reports_end_effector_pose(backend: PyBulletBackend) -> None:
    state = backend.get_end_effector_state("panda_arm")

    assert state.pose_stamped.header.frame_id == "world"
    assert state.pose_stamped.pose.orientation.w != 0.0


def test_pybullet_backend_loads_compiled_csd_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_pybullet.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    result = compile_csd_to_pybullet(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version="test-pybullet",
    )
    assert isinstance(result.manifest, CsdRealizationManifest)

    instance = PyBulletBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        sensors = {entry.name for entry in instance.list_sensors().entries}
        assert "world_camera" in sensors
        assert "mug" in instance.body_names
        assert "tray" in instance.body_names
        instance.set_object_pose(
            "mug",
            Pose(
                position=Point(x=0.12, y=0.0, z=0.4),
                orientation=Quaternion(w=1.0),
            ),
        )
        assert "mug" in instance.body_names
        image = instance.get_sensors(["world_camera"]).images[0]
        assert float(_image_array(image).std()) > 1.0
    finally:
        instance.shutdown()


def test_pybullet_backend_loads_and_runs_shared_openusd_csd(tmp_path: Path) -> None:
    registry = {
        "objects": [
            {
                "asset_id": asset_id,
                "backend_resources": [
                    {
                        "backend": "pybullet",
                        "resource_id": f"pybullet_{asset_id}",
                        "mesh_path": f"objects/{asset_id}.obj",
                        "resource_hash": f"hash_{asset_id}",
                    }
                ],
            }
            for asset_id in ("object_box", "object_anchor")
        ]
    }
    asset_root = tmp_path / "assets"
    _write_fixture_asset_files(asset_root, registry)
    result = compile_csd_to_pybullet(
        csd_path=SHARED_OPENUSD_CSD,
        asset_registry=registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version="test-pybullet",
    )
    assert result.blockers == ()
    assert result.manifest is not None

    instance = PyBulletBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        sensors = {entry.name for entry in instance.list_sensors().entries}
        image = instance.get_sensors(["Camera"]).images[0]
        time.sleep(0.05)

        assert "Camera" in sensors
        assert {"dynamic_box", "anchor", "table", "panda"} <= set(instance.body_names)
        assert float(_image_array(image).std()) > 1.0
        state = instance.get_robot_state()
        assert np.isfinite(state.position).all()
        assert np.isfinite(state.velocity).all()
    finally:
        instance.shutdown()


def test_pybullet_backend_loads_compiled_franka_csd_as_controllable_robot(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("franka_tabletop_single_object")
    asset_registry = _load_registry_fixture("asset_registry_pybullet.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    result = compile_csd_to_pybullet(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version="test-pybullet",
    )
    assert isinstance(result.manifest, CsdRealizationManifest)

    instance = PyBulletBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        spec = instance.get_robot_spec()
        assert spec.robot_name == "panda"
        assert "panda_arm" in {group.name for group in spec.joint_model_groups}
        assert "mug" in instance.body_names
        assert "panda" in instance.body_names
    finally:
        instance.shutdown()


def test_pybullet_backend_records_and_replays_lerobot_episode(
    tmp_path: Path,
    backend: PyBulletBackend,
) -> None:
    recorder = LerobotDataRecorder(tmp_path, backend)
    options = RecordOptions(
        repo_name="pybullet_demo",
        task_text="hold pose",
        fps=5,
        jmg_included=["panda_arm"],
        sensor_name_included=["world_camera"],
    )

    job = recorder.episode_start(options)
    time.sleep(0.25)
    end_status = recorder.episode_end()
    replay_status = recorder.episode_replay(RecordInfo(repo_name="pybullet_demo", episode_id=0))

    assert job.episode_id == 0
    assert end_status.code == 1
    assert replay_status.code == 1
