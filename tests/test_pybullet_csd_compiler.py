"""Tests for CSD -> PyBullet realization packages."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pybullet as p

from robosim.core import CsdRealizationManifest, compile_csd, compile_csd_to_pybullet

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"


def _load_json_fixture(name: str) -> dict[str, object]:
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


def test_compile_csd_to_pybullet_writes_self_contained_package(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_pybullet.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_pybullet(
        csd=csd,
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


def test_compile_csd_dispatches_to_pybullet(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_pybullet.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd(
        backend="pybullet",
        csd=csd,
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
