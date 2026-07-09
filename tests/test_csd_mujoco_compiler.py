"""Tests for the first CSD -> MuJoCo MJCF compiler."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from robosim.core import CsdRealizationManifest, compile_csd, compile_csd_to_mujoco


def test_compile_csd_to_mujoco_writes_loadable_mjcf_and_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    mesh_path = asset_root / "objects" / "mug.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
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
        output_root=tmp_path / "compiled",
        asset_root=asset_root,
        realization_version="test-0.1",
        simulator_version="test-mujoco",
    )
    manifest = result.manifest

    scene_path = tmp_path / "compiled" / "mujoco" / "csd_tabletop_0001" / "scene.xml"
    copied_mesh_path = (
        tmp_path / "compiled" / "mujoco" / "csd_tabletop_0001" / "assets" / "objects" / "mug.obj"
    )
    tree = ET.parse(scene_path)
    root = tree.getroot()

    assert isinstance(manifest, CsdRealizationManifest)
    assert result.blockers == ()
    assert manifest.csd_id == "csd_tabletop_0001"
    assert manifest.backend == "mujoco"
    assert manifest.entry_file == "scene.xml"
    assert manifest.generated_files == ("scene.xml", "assets/objects/mug.obj")
    assert copied_mesh_path.is_file()
    assert root.tag == "mujoco"
    assert root.find("compiler").attrib["meshdir"] == str(
        tmp_path / "compiled" / "mujoco" / "csd_tabletop_0001" / "assets"
    )
    assert root.find("asset/mesh").attrib == {
        "name": "object_mug",
        "file": "objects/mug.obj",
    }
    body = root.find("worldbody/body")
    assert body.attrib["name"] == "mug"
    assert body.attrib["pos"] == "0.25 -0.1 0.82"
    assert body.attrib["quat"] == "1 0 0 0"
    assert body.find("freejoint") is not None
    assert body.find("geom").attrib["mesh"] == "object_mug"

    mesh_path.unlink()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody >= 2


def test_compile_csd_rejects_gazebo_until_ros2_artifact_path_exists(tmp_path: Path) -> None:
    try:
        compile_csd(
            backend="gazebo",
            csd={"csd_id": "csd_gazebo"},
            asset_registry={},
            output_root=tmp_path,
            asset_root=tmp_path,
        )
    except NotImplementedError as exc:
        assert "Gazebo" in str(exc)
        assert "ROS2" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


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
