"""Tests for CSD -> PyBullet realization packages."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copytree
from typing import Mapping

import pybullet as p

from robosim.core import (
    CsdRealizationManifest,
    compile_csd,
    compile_csd_to_pybullet,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"
SHARED_OPENUSD_CSD = FIXTURE_ROOT / "openusd" / "shared_tabletop" / "csd.usda"
SEMANTIC_OPENUSD_ROOT = FIXTURE_ROOT / "openusd" / "semantic"


def _csd_fixture(name: str) -> Path:
    return SEMANTIC_OPENUSD_ROOT / name.removesuffix(".json") / "csd.usda"


def _load_registry_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _fixture_mesh_half_extents(path: Path) -> tuple[float, float, float]:
    name = path.stem
    if "tray" in name:
        return (0.08, 0.055, 0.012)
    if "marker" in name:
        return (0.018, 0.018, 0.055)
    if "can" in name:
        return (0.035, 0.035, 0.08)
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
        variants = record.get("backend_resources", ())
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            relative_path = variant.get("mesh_path") or variant.get("relative_path")
            if relative_path:
                _write_box_mesh(asset_root / str(relative_path))


def _load_generated_scene(scene_path: Path, client_id: int) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("generated_pybullet_scene", scene_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handles = module.load_scene(client_id)
    assert isinstance(handles, dict)
    return handles


def test_compile_csd_to_pybullet_consumes_composed_openusd_stage(tmp_path: Path) -> None:
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
    scene_root = Path(result.manifest.root_path)
    metadata = json.loads((scene_root / "scene_meta.json").read_text(encoding="utf-8"))
    assert metadata["csd_id"] == "csd_shared_tabletop"
    assert metadata["gravity"] == [0.0, 0.0, -9.81]
    assert metadata["objects"]["dynamic_box"]["mass_kg"] == 1.0
    assert metadata["objects"]["dynamic_box"]["friction"][0] == 0.7
    assert metadata["objects"]["dynamic_box"]["rgba"] == [0.8, 0.25, 0.15, 1.0]
    assert metadata["surfaces"]["table"]["half_extents"] == [0.6, 0.4, 0.05]
    assert metadata["robot"]["position"] == [-0.45, 0.0, 0.0]
    assert metadata["cameras"][0]["name"] == "Camera"

    urdf = ET.parse(scene_root / "assets" / "objects" / "dynamic_box.urdf").getroot()
    inertia = urdf.find("link/inertial/inertia")
    collision_mesh = urdf.find("link/collision/geometry/mesh")
    visual_color = urdf.find("link/visual/material/color")
    assert inertia is not None
    assert collision_mesh is not None
    assert visual_color is not None
    assert inertia.attrib == {
        "ixx": "0.015",
        "ixy": "0",
        "ixz": "0",
        "iyy": "0.015",
        "iyz": "0",
        "izz": "0.015",
    }
    assert collision_mesh.attrib["filename"] == "object_box.obj"
    assert visual_color.attrib["rgba"] == "0.8 0.25 0.15 1"


def test_compile_csd_to_pybullet_blocks_invalid_openusd_relationship(tmp_path: Path) -> None:
    csd_root = tmp_path / "invalid_relationship"
    copytree(SHARED_OPENUSD_CSD.parent, csd_root)
    task_layer = csd_root / "layers" / "task.usda"
    task_layer.write_text(
        task_layer.read_text(encoding="utf-8").replace(
            "</World/Objects/Anchor>",
            "</World/Objects/Missing>",
        ),
        encoding="utf-8",
    )

    result = compile_csd_to_pybullet(
        csd_path=csd_root / "csd.usda",
        asset_registry={"objects": []},
        output_root=tmp_path / "engine_manifests",
        asset_root=tmp_path / "assets",
        simulator_version="test-pybullet",
    )

    assert result.manifest is None
    assert len(result.blockers) == 1
    assert result.blockers[0].backend == "pybullet"
    assert "unresolved_relationship_target" in result.blockers[0].reason


def test_compile_csd_to_pybullet_writes_self_contained_package(tmp_path: Path) -> None:
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

    scene_root = tmp_path / "engine_manifests" / "pybullet" / "csd_object_only_0001"
    manifest_path = scene_root / "manifest.json"
    scene_path = scene_root / "scene.py"
    meta_path = scene_root / "scene_meta.json"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert result.manifest.backend == "pybullet"
    assert result.manifest.entry_file == "scene.py"
    assert manifest_path.is_file()
    assert scene_path.is_file()
    assert meta_path.is_file()
    assert (scene_root / "assets" / "objects" / "mug.obj").is_file()
    assert (scene_root / "assets" / "objects" / "tray.obj").is_file()
    assert (scene_root / "assets" / "objects" / "mug.urdf").is_file()
    assert (scene_root / "diagnostics" / "load_check.json").is_file()
    assert (scene_root / "diagnostics" / "physics_check.json").is_file()
    assert (scene_root / "diagnostics" / "semantic_preview.ppm").is_file()

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["backend"] == "pybullet"
    assert metadata["csd_id"] == "csd_object_only_0001"
    assert metadata["objects"]["mug"]["asset_id"] == "object_mug"
    assert metadata["objects"]["mug"]["urdf_path"] == "assets/objects/mug.urdf"
    assert metadata["cameras"][0]["name"] == "world_camera"

    (asset_root / "objects" / "mug.obj").unlink()
    client_id = p.connect(p.DIRECT)
    try:
        handles = _load_generated_scene(scene_path, client_id)
        bodies = handles["bodies"]
        assert isinstance(bodies, dict)
        assert {"mug", "tray"} <= set(bodies)
        mug_pos, mug_orn = p.getBasePositionAndOrientation(
            int(bodies["mug"]),
            physicsClientId=client_id,
        )
        assert mug_pos == pytest_approx_tuple((0.05, 0.0, 0.32))
        assert mug_orn == pytest_approx_tuple((0.0, 0.0, 0.0, 1.0))
    finally:
        p.disconnect(client_id)


def test_compile_csd_to_pybullet_includes_franka_robot(tmp_path: Path) -> None:
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

    scene_root = tmp_path / "engine_manifests" / "pybullet" / "csd_tabletop_0001"
    metadata = json.loads((scene_root / "scene_meta.json").read_text(encoding="utf-8"))

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert metadata["robot_name"] == "panda"
    assert metadata["robot"]["asset_id"] == "robot_franka_panda"
    assert metadata["robot"]["urdf_path"] == "assets/robots/franka_panda/panda.urdf"
    assert (scene_root / "assets" / "robots" / "franka_panda" / "panda.urdf").is_file()
    assert any(
        path.startswith("assets/robots/franka_panda/meshes/")
        for path in result.manifest.generated_files
    )

    client_id = p.connect(p.DIRECT)
    try:
        handles = _load_generated_scene(scene_root / "scene.py", client_id)
        bodies = handles["bodies"]
        assert isinstance(bodies, dict)
        assert {"panda", "mug"} <= set(bodies)
        assert p.getNumJoints(int(bodies["panda"]), physicsClientId=client_id) > 0
    finally:
        p.disconnect(client_id)


def test_compile_csd_dispatches_to_pybullet(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_pybullet.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd(
        backend="pybullet",
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version="test-pybullet",
    )

    assert isinstance(result.manifest, CsdRealizationManifest)
    assert result.manifest.backend == "pybullet"


def pytest_approx_tuple(values: tuple[float, ...]) -> tuple[object, ...]:
    import pytest

    return tuple(pytest.approx(value, abs=1e-6) for value in values)
