"""Compile CSD artifacts into a narrow MuJoCo MJCF scene."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robosim.core.csd import (
    CsdRealizationBlocker,
    CsdRealizationManifest,
    asset_variant_hashes_for_csd,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)

BACKEND = "mujoco"
DEFAULT_REALIZATION_VERSION = "mujoco-csd-compiler-0.1"


@dataclass(frozen=True, slots=True)
class CsdMujocoCompilationResult:
    """Result of compiling one CSD into MuJoCo artifacts."""

    manifest: CsdRealizationManifest | None
    blockers: tuple[CsdRealizationBlocker, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(self.blockers))


def compile_csd_to_mujoco(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    output_root: Path,
    asset_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdMujocoCompilationResult:
    """Compile a fixed CSD into a minimal MuJoCo MJCF scene.

    This first compiler supports rigid mesh objects with explicit CSD poses. It
    uses MJCF `compiler meshdir`, `asset/mesh file`, and mesh geoms as described
    in the MuJoCo XML Reference. Backend load/render validation remains a later
    runtime step.
    """
    blockers = find_csd_realization_blockers(
        csd=csd,
        asset_registry=asset_registry,
        backend=BACKEND,
    )
    if blockers:
        return CsdMujocoCompilationResult(manifest=None, blockers=blockers)

    variants = _variants_by_asset(asset_registry)
    mesh_blockers = _mesh_path_blockers(csd, variants, Path(asset_root))
    if mesh_blockers:
        return CsdMujocoCompilationResult(manifest=None, blockers=mesh_blockers)

    csd_id = _required_str(csd, "csd_id")
    realization_config = dict(realization_config or {})
    variant_hashes = asset_variant_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=BACKEND,
    )
    cache_key = make_csd_realization_cache_key(
        csd=csd,
        asset_variant_hashes=variant_hashes,
        backend=BACKEND,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )
    scene_root = Path(output_root) / BACKEND / csd_id
    scene_root.mkdir(parents=True, exist_ok=True)
    scene_path = scene_root / "scene.xml"
    _write_mjcf(
        scene_path,
        csd=csd,
        asset_root=Path(asset_root),
        variants=variants,
    )
    return CsdMujocoCompilationResult(
        manifest=CsdRealizationManifest(
            manifest_id=f"manifest_{BACKEND}_{csd_id}",
            csd_id=csd_id,
            backend=BACKEND,
            cache_key=cache_key.digest,
            root_path=str(scene_root),
            entry_file="scene.xml",
            generated_files=("scene.xml",),
            preview_files=(),
        )
    )


def _write_mjcf(
    scene_path: Path,
    *,
    csd: Mapping[str, Any],
    asset_root: Path,
    variants: Mapping[str, Mapping[str, Any]],
) -> None:
    root = ET.Element("mujoco", {"model": _required_str(csd, "csd_id")})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "coordinate": "local",
            "meshdir": str(asset_root),
        },
    )
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    assets = ET.SubElement(root, "asset")
    for asset_id in _csd_asset_ids(csd):
        ET.SubElement(
            assets,
            "mesh",
            {
                "name": _mjcf_name(asset_id),
                "file": str(variants[asset_id]["relative_path"]),
            },
        )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key_light", "pos": "0 -1 3", "dir": "0 0 -1"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "2 2 0.02",
            "rgba": "0.8 0.8 0.8 1",
        },
    )
    for obj in _csd_objects(csd):
        _append_object_body(worldbody, obj)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)


def _append_object_body(parent: ET.Element, obj: Mapping[str, Any]) -> None:
    asset_id = _required_str(obj, "asset_id")
    body = ET.SubElement(
        parent,
        "body",
        {
            "name": _mjcf_name(_required_str(obj, "name")),
            "pos": _vector_text(obj, "position"),
            "quat": _quaternion_text(obj),
        },
    )
    if not bool(obj.get("static", False)):
        ET.SubElement(body, "freejoint")
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{_mjcf_name(_required_str(obj, 'name'))}_geom",
            "type": "mesh",
            "mesh": _mjcf_name(asset_id),
            "mass": _object_scalar(obj, "mass_kg", 0.1),
            "friction": _object_scalar(obj, "friction", 0.7),
            "rgba": "0.7 0.7 0.7 1",
        },
    )


def _mesh_path_blockers(
    csd: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    asset_root: Path,
) -> tuple[CsdRealizationBlocker, ...]:
    csd_id = _required_str(csd, "csd_id")
    blockers: list[CsdRealizationBlocker] = []
    for asset_id in _csd_asset_ids(csd):
        relative_path = str(variants[asset_id].get("relative_path", ""))
        if not relative_path:
            blockers.append(_asset_blocker(csd_id, asset_id, "asset variant has no relative_path"))
            continue
        if Path(relative_path).is_absolute():
            blockers.append(_asset_blocker(csd_id, asset_id, "asset variant path must be relative"))
            continue
        if not (asset_root / relative_path).is_file():
            blockers.append(
                _asset_blocker(csd_id, asset_id, f"asset variant file is missing: {relative_path}")
            )
    return tuple(blockers)


def _asset_blocker(csd_id: str, asset_id: str, reason: str) -> CsdRealizationBlocker:
    return CsdRealizationBlocker(
        blocker_id=f"{csd_id}_{BACKEND}_{asset_id}_compile_blocked",
        csd_id=csd_id,
        backend=BACKEND,
        asset_id=asset_id,
        scope="asset",
        reason=reason,
    )


def _variants_by_asset(asset_registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    variants: dict[str, Mapping[str, Any]] = {}
    for record in (*asset_registry.get("objects", ()), *asset_registry.get("assets", ())):
        if not isinstance(record, Mapping):
            continue
        asset_id = str(record.get("object_id") or record.get("asset_id") or "")
        if asset_id and asset_id not in variants:
            variant = _passed_mujoco_variant(record.get("variants", ()))
            if variant is not None:
                variants[asset_id] = variant
    return variants


def _passed_mujoco_variant(variant_records: object) -> Mapping[str, Any] | None:
    if not isinstance(variant_records, (list, tuple)):
        return None
    for variant in variant_records:
        if not isinstance(variant, Mapping):
            continue
        if (
            str(variant.get("engine", "")) == BACKEND
            and str(variant.get("validation_state", "")).lower() == "passed"
        ):
            return variant
    return None


def _csd_objects(csd: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(obj for obj in csd.get("objects", ()) if isinstance(obj, Mapping))


def _csd_asset_ids(csd: Mapping[str, Any]) -> tuple[str, ...]:
    asset_ids = [str(obj["asset_id"]) for obj in _csd_objects(csd) if obj.get("asset_id")]
    return tuple(dict.fromkeys(asset_ids))


def _vector_text(obj: Mapping[str, Any], key: str) -> str:
    position = _pose_part(obj, key)
    return _numbers_text((position["x"], position["y"], position["z"]))


def _quaternion_text(obj: Mapping[str, Any]) -> str:
    orientation = _pose_part(obj, "orientation")
    return _numbers_text((orientation["w"], orientation["x"], orientation["y"], orientation["z"]))


def _pose_part(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    pose = obj.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("CSD object pose is required for MuJoCo compilation")
    part = pose.get(key)
    if not isinstance(part, Mapping):
        raise ValueError(f"CSD object pose.{key} is required for MuJoCo compilation")
    return part


def _object_scalar(obj: Mapping[str, Any], key: str, default: float) -> str:
    initial_state = obj.get("initial_state", {})
    if isinstance(initial_state, Mapping) and key in initial_state:
        return _number_text(initial_state[key])
    return _number_text(default)


def _numbers_text(values: tuple[object, ...]) -> str:
    return " ".join(_number_text(value) for value in values)


def _number_text(value: Any) -> str:
    return f"{float(value):g}"


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _mjcf_name(value: str) -> str:
    name = "_".join(str(value).strip().replace("-", "_").split())
    return name or "unnamed"
