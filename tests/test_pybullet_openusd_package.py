"""Tests for v9 OpenUSD resource-package realization in PyBullet."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pybullet as p

from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.pybullet.backend import PyBulletBackend
from robosim.core.csd_compiler import compile_csd
from robosim.core.mujoco_openusd_package import read_openusd_scene_package

EXPERIMENT_SCENE = Path(__file__).parents[1] / "example" / "experiment" / "scene.usda"
EXPERIMENT_9D_SCENE = Path(__file__).parents[1] / "example" / "experiment-9d" / "scene.usda"


def test_compile_v9_openusd_package_to_pybullet_and_load_runtime(tmp_path: Path) -> None:
    result = compile_csd(
        backend="pybullet",
        csd_path=EXPERIMENT_SCENE,
        output_root=tmp_path / "engine_manifests",
        simulator_version="test-pybullet",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    root = Path(result.manifest.root_path)
    metadata = json.loads((root / "scene_meta.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "pybullet"
    assert metadata["robot_name"] == "robot"
    assert metadata["light"] == {"direction": [0.0, 0.0, -1.0], "intensity": 1.0}
    assert {item["name"] for item in metadata["instances"]} == {
        "room_4x4_empty",
        "cabinet_double_door_01",
        "stool_square_low_01",
    }
    assert (root / "assets").is_dir()
    assert (root / "diagnostics" / "preview.ppm").is_file()
    assert result.manifest.preview_files == ("diagnostics/preview.ppm",)
    assert (root / "robots" / "franka_panda" / "panda.urdf").is_file()
    cabinet = next(
        item for item in metadata["instances"] if item["name"] == "cabinet_double_door_01"
    )
    assert cabinet["dynamic_friction"] == 0.5
    assert cabinet["restitution"] == 0.0
    material_files = list((root / "assets").glob("**/visual.mtl"))
    assert len(material_files) >= 6
    assert any("map_Kd" in path.read_text(encoding="utf-8") for path in material_files)
    package = read_openusd_scene_package(EXPERIMENT_SCENE)
    source = next(
        item.asset
        for item in package.instances
        if item.prim_path.endswith("cabinet_double_door_01")
    )
    child_links = {joint.child for joint in source.joints}
    root_body = next(body for body in source.bodies if body.path not in child_links)
    urdf = ET.parse(root / cabinet["urdf_path"])
    inertial = urdf.find(f"link[@name='{root_body.path}']/inertial/origin")
    assert inertial is not None
    assert inertial.attrib["xyz"] == " ".join(str(value) for value in root_body.center_of_mass)

    backend = PyBulletBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        assert {"robot", "room_4x4_empty", "cabinet_double_door_01", "stool_square_low_01"} <= set(
            backend._body_names
        )
        assert p.getNumJoints(backend._body_names["cabinet_double_door_01"]) > 0
        dynamics = p.getDynamicsInfo(backend._body_names["cabinet_double_door_01"], -1)
        assert dynamics[1] == 0.5
    finally:
        backend.shutdown()


def test_compile_v9d_shell_with_separate_floor_and_wall_visuals(tmp_path: Path) -> None:
    result = compile_csd(
        backend="pybullet",
        csd_path=EXPERIMENT_9D_SCENE,
        output_root=tmp_path / "engine_manifests",
        simulator_version="test-pybullet",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    root = Path(result.manifest.root_path)
    shell_urdf = next(
        path
        for path in (root / "assets").glob("*/asset.urdf")
        if {link.attrib["name"] for link in ET.parse(path).findall("link")} >= {"Floor", "Walls"}
    )
    assert shell_urdf.is_file()
    assert list((root / "assets").glob("**/floor/visual.mtl"))
    assert list((root / "assets").glob("**/walls/visual.mtl"))

    backend = PyBulletBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        assert backend.robot_name == "robot"
        state = backend.get_robot_state()
        assert "panda_joint1" in state.name
        assert "panda_arm" in {group.name for group in backend.get_robot_spec().joint_model_groups}
        backend.set_joint_target(
            ["panda_joint1"],
            [0.1],
            core_pb2.JointCommand.ControlMode.POSITION,
            group="panda_arm",
        )
        command_state = backend.get_joint_command_state()
        assert command_state.position[list(state.name).index("panda_joint1")] == 0.1
        assert "agent_view" in {entry.name for entry in backend.list_sensors().entries}
        image = backend.get_sensors(["agent_view"]).images[0]
        assert image.name == "agent_view"
        assert len(image.data) == image.width * image.height * 3
    finally:
        backend.shutdown()
