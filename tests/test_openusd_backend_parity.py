"""Cross-backend realization acceptance for portable OpenUSD CSD scenes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from robosim.core import compile_csd
from robosim.core.openusd_csd import compiler_csd_from_openusd, read_openusd_csd

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"
OPENUSD_ROOT = FIXTURE_ROOT / "openusd"
PORTABLE_CSD_CASES = {
    "shared_tabletop": OPENUSD_ROOT / "shared_tabletop" / "csd.usda",
    "object_only_static_and_dynamic": OPENUSD_ROOT
    / "semantic"
    / "object_only_static_and_dynamic"
    / "csd.usda",
    "tabletop_rotated_surface_object": OPENUSD_ROOT
    / "semantic"
    / "tabletop_rotated_surface_object"
    / "csd.usda",
    "low_gravity_static_layout": OPENUSD_ROOT
    / "semantic"
    / "low_gravity_static_layout"
    / "csd.usda",
}
BACKENDS = ("mujoco", "pybullet", "gazebo")


def _load_registry(backend: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / f"asset_registry_{backend}.json").read_text(encoding="utf-8"))


def _fixture_mesh_half_extents(path: Path) -> tuple[float, float, float]:
    name = path.stem
    if name in {"box", "object_box"}:
        return (0.15, 0.15, 0.15)
    if name in {"anchor", "object_anchor"}:
        return (0.1, 0.1, 0.1)
    if "tray" in name:
        return (0.08, 0.055, 0.012)
    if "marker" in name:
        return (0.018, 0.018, 0.055)
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


def _registry_for_csd(csd_path: Path, backend: str) -> dict[str, object]:
    registry = _load_registry(backend)
    records = registry["objects"]
    assert isinstance(records, list)
    existing_asset_ids = {
        str(record["asset_id"])
        for record in records
        if isinstance(record, Mapping) and "asset_id" in record
    }
    csd = compiler_csd_from_openusd(read_openusd_csd(csd_path, backend=backend))
    for obj in csd.objects:
        if obj.asset_id in existing_asset_ids:
            continue
        records.append(
            {
                "asset_id": obj.asset_id,
                "backend_resources": [
                    {
                        "backend": backend,
                        "resource_id": f"{backend}_{obj.asset_id}",
                        "mesh_path": f"objects/{obj.asset_id}.obj",
                        "resource_hash": f"hash_{backend}_{obj.asset_id}",
                    }
                ],
            }
        )
    return registry


def _write_registry_assets(asset_root: Path, registry: Mapping[str, object]) -> None:
    records = registry.get("objects", ())
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, Mapping)
        resources = record.get("backend_resources", ())
        assert isinstance(resources, list)
        for resource in resources:
            assert isinstance(resource, Mapping)
            _write_box_mesh(asset_root / str(resource["mesh_path"]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "csd_path",
    PORTABLE_CSD_CASES.values(),
    ids=PORTABLE_CSD_CASES.keys(),
)
def test_portable_openusd_scene_realizes_in_each_backend(
    tmp_path: Path,
    csd_path: Path,
    backend: str,
) -> None:
    registry = _registry_for_csd(csd_path, backend)
    asset_root = tmp_path / "assets"
    _write_registry_assets(asset_root, registry)

    result = compile_csd(
        backend=backend,
        csd_path=csd_path,
        asset_registry=registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version=f"test-{backend}",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    scene_root = Path(result.manifest.root_path)
    assert (scene_root / result.manifest.entry_file).is_file()
    for relative_path in (*result.manifest.generated_files, *result.manifest.preview_files):
        assert (scene_root / relative_path).is_file()

    if backend in {"mujoco", "pybullet"}:
        assert result.manifest.preview_files == ("diagnostics/semantic_preview.ppm",)
        for relative_path in (
            "diagnostics/load_check.json",
            "diagnostics/physics_check.json",
        ):
            assert (scene_root / relative_path).is_file()
    else:
        assert result.manifest.preview_files == ()
        for relative_path in (
            "diagnostics/sdf_check.json",
            "diagnostics/headless_load.json",
        ):
            payload = json.loads((scene_root / relative_path).read_text(encoding="utf-8"))
            assert payload["status"] == "passed"
