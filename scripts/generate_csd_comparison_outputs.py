"""Generate comparable MuJoCo/PyBullet CSD realization artifacts under /tmp."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from robosim.core import (  # noqa: E402
    CsdCompilationResult,
    CsdRealizationManifest,
    compile_csd_to_mujoco,
    compile_csd_to_pybullet,
)

FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "csd"
DEFAULT_OUTPUT_ROOT = Path("/tmp/robosim-csd-compiler-comparison")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile selected CSD fixtures for MuJoCo and PyBullet into comparable dirs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        help="Fixture JSON name. Defaults to every non-registry CSD fixture.",
    )
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    fixtures = tuple(args.fixtures or _all_csd_fixture_names())
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_root": str(output_root),
        "asset_mode": (
            "synthetic_fixture_assets: object meshes/textures are generated under "
            "fixture_assets from the test asset registries, not copied from production assets"
        ),
        "fixtures": [
            _compile_fixture(output_root, fixture_name)
            for fixture_name in fixtures
        ],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


def _all_csd_fixture_names() -> tuple[str, ...]:
    return tuple(
        path.name
        for path in sorted(FIXTURE_ROOT.glob("*.json"))
        if not path.name.startswith("asset_registry_")
    )


def _compile_fixture(output_root: Path, fixture_name: str) -> dict[str, object]:
    csd = _load_json(FIXTURE_ROOT / fixture_name)
    return {
        "fixture": fixture_name,
        "mujoco": _compile_backend(
            backend="mujoco",
            csd=csd,
            output_root=output_root,
            compiler=compile_csd_to_mujoco,
        ),
        "pybullet": _compile_backend(
            backend="pybullet",
            csd=csd,
            output_root=output_root,
            compiler=compile_csd_to_pybullet,
        ),
    }


def _compile_backend(
    *,
    backend: str,
    csd: Mapping[str, Any],
    output_root: Path,
    compiler: Callable[..., CsdCompilationResult],
) -> dict[str, object]:
    asset_registry = _load_json(FIXTURE_ROOT / f"asset_registry_{backend}.json")
    asset_root = output_root / backend / "fixture_assets"
    _write_fixture_asset_files(asset_root, asset_registry)
    result = compiler(
        csd=csd,
        asset_registry=asset_registry,
        output_root=output_root / backend / "engine_manifests",
        asset_root=asset_root,
        simulator_version=f"comparison-{backend}",
    )
    if result.manifest is None:
        return {
            "blockers": [blocker.to_json_dict() for blocker in result.blockers],
        }
    manifest = result.manifest
    return {
        "manifest": manifest.to_json_dict(),
        "root_path": manifest.root_path,
        "entry_file": str(Path(manifest.root_path) / manifest.entry_file),
        "preview_files": _preview_paths(manifest),
    }


def _preview_paths(manifest: CsdRealizationManifest) -> list[str]:
    return [str(Path(manifest.root_path) / path) for path in manifest.preview_files]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_png_1x1(path: Path) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"\x00\xff\xff\xff\xff"
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_fixture_asset_files(asset_root: Path, asset_registry: Mapping[str, Any]) -> None:
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
            collision_mesh_path = variant.get("collision_mesh_path")
            if collision_mesh_path:
                _write_box_mesh(asset_root / str(collision_mesh_path))
            material = variant.get("material")
            if isinstance(material, Mapping) and material.get("texture_path"):
                _write_png_1x1(asset_root / str(material["texture_path"]))


if __name__ == "__main__":
    main()
