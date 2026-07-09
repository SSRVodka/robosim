"""Tests for CSD -> backend-native scene compilers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copytree

import mujoco

from robosim.core import (
    CsdRealizationManifest,
    compile_csd,
    compile_csd_to_gazebo,
    compile_csd_to_mujoco,
)


def _write_tetra_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 0.04 0 0",
                "v 0 0.04 0",
                "v 0 0 0.04",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            )
        ),
        encoding="utf-8",
    )


def test_compile_csd_to_mujoco_writes_loadable_mjcf_and_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    mesh_path = asset_root / "objects" / "mug.obj"
    _write_tetra_mesh(mesh_path)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)
    csd = {
        "csd_id": "csd_tabletop_0001",
        "task_instance_id": "task_tabletop_0001",
        "task_objective": "place the mug on the table",
        "frame": "world",
        "units": "m",
        "robot_asset_id": "robot_franka_panda",
        "world_template_id": "world_tabletop",
        "objects": [
            {
                "name": "mug",
                "asset_id": "object_mug",
                "role": "interactive_object",
                "pose": {
                    "position": {"x": 0.25, "y": -0.1, "z": 0.82},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "static": False,
                "initial_state": {"mass_kg": 0.2, "friction": 0.8},
            }
        ],
        "relationships": [],
        "randomization": {"seed": 3, "values": {}},
        "sensor_requirements": {},
        "evaluator_refs": [],
        "schema_version": "0.1",
    }
    asset_registry = {
        "objects": [
            {
                "object_id": "object_mug",
                "variants": [
                    {
                        "engine": "mujoco",
                        "relative_path": "objects/mug.obj",
                        "variant_hash": "hash_mug_obj",
                        "validation_state": "passed",
                    }
                ],
            }
        ]
    }

    result = compile_csd(
        backend="mujoco",
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        realization_config={"robot_template_dir": str(template_copy)},
        realization_version="test-0.1",
        simulator_version="test-mujoco",
    )
    manifest = result.manifest

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_tabletop_0001"
    scene_path = scene_root / "scene.xml"
    manifest_path = scene_root / "manifest.json"
    copied_mesh_path = (
        scene_root / "assets" / "objects" / "mug.obj"
    )
    copied_robot_xml = scene_root / "assets" / "robots" / "franka_panda" / "panda.xml"
    copied_robot_srdf = scene_root / "assets" / "robots" / "franka_panda" / "panda.srdf"
    copied_robot_mesh = scene_root / "assets" / "link0.stl"
    tree = ET.parse(scene_path)
    root = tree.getroot()

    assert isinstance(manifest, CsdRealizationManifest)
    assert result.blockers == ()
    assert manifest.csd_id == "csd_tabletop_0001"
    assert manifest.backend == "mujoco"
    assert manifest.entry_file == "scene.xml"
    assert "manifest.json" in manifest.generated_files
    assert "scene.xml" in manifest.generated_files
    assert "assets/objects/mug.obj" in manifest.generated_files
    assert "assets/robots/franka_panda/panda.xml" in manifest.generated_files
    assert copied_mesh_path.is_file()
    assert copied_robot_xml.is_file()
    assert copied_robot_srdf.is_file()
    assert copied_robot_mesh.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest.to_json_dict()
    assert root.tag == "mujoco"
    assert root.find("compiler") is None
    assert root.find("include").attrib["file"] == "assets/robots/franka_panda/panda.xml"
    assert root.find("asset/mesh").attrib == {
        "name": "object_mug",
        "file": "objects/mug.obj",
    }
    assert root.find("worldbody/camera").attrib["name"] == "world_camera"
    body = root.find("worldbody/body")
    assert body.attrib["name"] == "mug"
    assert body.attrib["pos"] == "0.25 -0.1 0.82"
    assert body.attrib["quat"] == "1 0 0 0"
    assert body.find("freejoint") is not None
    assert body.find("geom").attrib["mesh"] == "object_mug"

    mesh_path.unlink()
    copied_template_source = template_copy
    for source_file in copied_template_source.rglob("*"):
        if source_file.is_file():
            source_file.unlink()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody >= 2


def test_compile_csd_to_mujoco_reports_missing_variant(tmp_path: Path) -> None:
    result = compile_csd_to_mujoco(
        csd={
            "csd_id": "csd_missing",
            "objects": [{"name": "mug", "asset_id": "object_mug"}],
        },
        asset_registry={"objects": [{"object_id": "object_mug", "variants": []}]},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_missing_mujoco_object_mug_variant_missing",
        "csd_id": "csd_missing",
        "backend": "mujoco",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "asset has no passed backend variant for mujoco",
    }


def test_compile_csd_to_mujoco_reports_unsupported_robot_asset(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    mesh_path = asset_root / "objects" / "mug.obj"
    _write_tetra_mesh(mesh_path)

    result = compile_csd_to_mujoco(
        csd={
            "csd_id": "csd_unknown_robot",
            "robot_asset_id": "robot_unknown",
            "objects": [
                {
                    "name": "mug",
                    "asset_id": "object_mug",
                    "pose": {
                        "position": {"x": 0.25, "y": -0.1, "z": 0.82},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                }
            ],
        },
        asset_registry={
            "objects": [
                {
                    "object_id": "object_mug",
                    "variants": [
                        {
                            "engine": "mujoco",
                            "relative_path": "objects/mug.obj",
                            "variant_hash": "hash_mug_obj",
                            "validation_state": "passed",
                        }
                    ],
                }
            ]
        },
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_unknown_robot_mujoco_robot_unknown_compile_blocked",
        "csd_id": "csd_unknown_robot",
        "backend": "mujoco",
        "asset_id": "robot_unknown",
        "scope": "asset",
        "reason": "no MuJoCo robot template is configured for robot asset",
    }


def test_compile_csd_to_gazebo_writes_self_contained_sdf_world(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    mesh_path = asset_root / "objects" / "mug.obj"
    _write_tetra_mesh(mesh_path)
    csd = {
        "csd_id": "csd_tabletop_0001",
        "objects": [
            {
                "name": "mug",
                "asset_id": "object_mug",
                "pose": {
                    "position": {"x": 0.25, "y": -0.1, "z": 0.82},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "static": False,
                "initial_state": {"mass_kg": 0.2, "friction": 0.8},
            }
        ],
    }
    asset_registry = {
        "objects": [
            {
                "object_id": "object_mug",
                "variants": [
                    {
                        "engine": "gazebo",
                        "relative_path": "objects/mug.obj",
                        "variant_hash": "hash_mug_obj",
                        "validation_state": "passed",
                    }
                ],
            }
        ]
    }

    result = compile_csd(
        backend="gazebo",
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "compiled",
        asset_root=asset_root,
        realization_version="test-0.1",
        simulator_version="test-gazebo",
    )

    manifest = result.manifest
    world_root = tmp_path / "compiled" / "gazebo" / "csd_tabletop_0001"
    world_path = world_root / "world.sdf"
    manifest_path = world_root / "manifest.json"
    copied_mesh_path = world_root / "assets" / "objects" / "mug.obj"
    root = ET.parse(world_path).getroot()
    model = root.find("world/model")

    assert isinstance(manifest, CsdRealizationManifest)
    assert result.blockers == ()
    assert manifest.backend == "gazebo"
    assert manifest.entry_file == "world.sdf"
    assert manifest.generated_files == ("manifest.json", "world.sdf", "assets/objects/mug.obj")
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest.to_json_dict()
    assert copied_mesh_path.is_file()
    assert root.attrib["version"] == "1.12"
    assert model.attrib["name"] == "mug"
    assert model.find("pose").text == "0.25 -0.1 0.82 0 0 0"
    assert model.find("static").text == "false"
    assert model.find("link/visual/geometry/mesh/uri").text == "assets/objects/mug.obj"
    assert model.find("link/collision/geometry/mesh/uri").text == "assets/objects/mug.obj"

    mesh_path.unlink()
    assert copied_mesh_path.is_file()


def test_compile_csd_to_gazebo_reports_missing_variant(tmp_path: Path) -> None:
    result = compile_csd_to_gazebo(
        csd={
            "csd_id": "csd_missing",
            "objects": [{"name": "mug", "asset_id": "object_mug"}],
        },
        asset_registry={"objects": [{"object_id": "object_mug", "variants": []}]},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_missing_gazebo_object_mug_variant_missing",
        "csd_id": "csd_missing",
        "backend": "gazebo",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "asset has no passed backend variant for gazebo",
    }
