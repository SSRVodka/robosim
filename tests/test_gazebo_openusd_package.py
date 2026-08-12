"""Tests for v9 OpenUSD resource-package realization in Gazebo."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from robosim.core.csd_compiler import compile_csd

EXPERIMENT_9D_SCENE = Path(__file__).parents[1] / "example" / "experiment-9d" / "scene.usda"


def test_compile_v9d_openusd_package_to_self_contained_gazebo_world(tmp_path: Path) -> None:
    result = compile_csd(
        backend="gazebo",
        csd_path=EXPERIMENT_9D_SCENE,
        output_root=tmp_path / "engine_manifests",
        simulator_version="test-gazebo",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    root = Path(result.manifest.root_path)
    world = ET.parse(root / "world.sdf").getroot().find("world")
    assert world is not None
    assert ET.parse(root / "world.sdf").getroot().attrib["version"] == "1.7"
    models = {model.attrib["name"]: model for model in world.findall("model")}
    assert {
        "robot",
        "room_4x4_empty_square",
        "cabinet_double_door_01",
        "stool_square_low_01",
    } <= set(models)
    cabinet = models["cabinet_double_door_01"]
    assert cabinet.findall("joint")
    assert len(cabinet.findall(".//collision")) > 1
    assert len(cabinet.findall(".//visual")) > len(cabinet.findall("link"))
    assert len({item.text for item in cabinet.findall(".//visual/material/diffuse")}) > 1
    assert world.find("state/model[@name='cabinet_double_door_01']/joint/angle") is not None
    shell = models["room_4x4_empty_square"]
    assert {link.attrib["name"] for link in shell.findall("link")} == {"Floor", "Walls"}
    assert all(link.find("inertial") is None for link in shell.findall("link"))
    mesh_uris = [Path(str(uri.text)) for uri in world.findall(".//mesh/uri")]
    assert mesh_uris
    assert all(not uri.is_absolute() and (root / uri).is_file() for uri in mesh_uris)
    assert (root / "diagnostics" / "sdf_check.json").is_file()
    assert (root / "diagnostics" / "entity_mapping.json").is_file()
