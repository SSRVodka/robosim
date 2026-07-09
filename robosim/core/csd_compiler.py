"""Compile fixed CSD artifacts into backend-native scene files."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any, Mapping

from robosim.core.csd import (
    CsdRealizationBlocker,
    CsdRealizationManifest,
    asset_variant_hashes_for_csd,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)

MUJOCO_BACKEND = "mujoco"
GAZEBO_BACKEND = "gazebo"
DEFAULT_REALIZATION_VERSION = "csd-compiler-0.2"


@dataclass(frozen=True, slots=True)
class CsdCompilationResult:
    """Result of compiling one fixed CSD into backend-native artifacts."""

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
) -> CsdCompilationResult:
    """Compile a fixed CSD into a minimal MuJoCo MJCF scene.

    This first compiler supports rigid mesh objects with explicit CSD poses. It
    uses MJCF `compiler meshdir`, `asset/mesh file`, and mesh geoms as described
    in the MuJoCo XML Reference. Backend load/render validation remains a later
    runtime step.
    """
    blockers = find_csd_realization_blockers(
        csd=csd,
        asset_registry=asset_registry,
        backend=MUJOCO_BACKEND,
    )
    if blockers:
        return CsdCompilationResult(manifest=None, blockers=blockers)

    variants = _variants_by_asset(asset_registry, backend=MUJOCO_BACKEND)
    mesh_blockers = _mesh_path_blockers(
        csd,
        variants,
        Path(asset_root),
        backend=MUJOCO_BACKEND,
    )
    if mesh_blockers:
        return CsdCompilationResult(manifest=None, blockers=mesh_blockers)

    csd_id = _required_str(csd, "csd_id")
    realization_config = dict(realization_config or {})
    robot_blockers = _mujoco_robot_template_blockers(
        csd=csd,
        realization_config=realization_config,
    )
    if robot_blockers:
        return CsdCompilationResult(manifest=None, blockers=robot_blockers)

    variant_hashes = asset_variant_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=MUJOCO_BACKEND,
    )
    cache_key = make_csd_realization_cache_key(
        csd=csd,
        asset_variant_hashes=variant_hashes,
        backend=MUJOCO_BACKEND,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )
    scene_root = Path(output_root) / MUJOCO_BACKEND / csd_id
    compiled_asset_root = scene_root / "assets"
    diagnostics_root = scene_root / "diagnostics"
    scene_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(exist_ok=True)
    generated_asset_files = _copy_variant_files(
        csd=csd,
        variants=variants,
        source_asset_root=Path(asset_root),
        compiled_asset_root=compiled_asset_root,
    )
    robot_include, generated_robot_files = _copy_mujoco_robot_template(
        csd=csd,
        realization_config=realization_config,
        scene_root=scene_root,
        compiled_asset_root=compiled_asset_root,
    )
    scene_path = scene_root / "scene.xml"
    _write_mjcf(
        scene_path,
        csd=csd,
        asset_root=compiled_asset_root,
        variants=variants,
        robot_include=robot_include,
    )
    generated_files = (
        "manifest.json",
        "scene.xml",
        *generated_asset_files,
        *generated_robot_files,
    )
    manifest = CsdRealizationManifest(
        manifest_id=f"manifest_{MUJOCO_BACKEND}_{csd_id}",
        csd_id=csd_id,
        backend=MUJOCO_BACKEND,
        cache_key=cache_key.digest,
        root_path=str(scene_root),
        entry_file="scene.xml",
        generated_files=_unique_files(generated_files),
        preview_files=(),
    )
    _write_manifest(scene_root / "manifest.json", manifest)
    return CsdCompilationResult(
        manifest=manifest
    )


def compile_csd(
    *,
    backend: str,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    output_root: Path,
    asset_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile one CSD for a requested backend target."""
    backend_key = backend.strip().lower()
    if backend_key == MUJOCO_BACKEND:
        return compile_csd_to_mujoco(
            csd=csd,
            asset_registry=asset_registry,
            output_root=output_root,
            asset_root=asset_root,
            realization_config=realization_config,
            realization_version=realization_version,
            simulator_version=simulator_version,
        )
    if backend_key == GAZEBO_BACKEND:
        return compile_csd_to_gazebo(
            csd=csd,
            asset_registry=asset_registry,
            output_root=output_root,
            asset_root=asset_root,
            realization_config=realization_config,
            realization_version=realization_version,
            simulator_version=simulator_version,
        )
    raise ValueError(f"unsupported CSD compiler backend: {backend}")


def compile_csd_to_gazebo(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    output_root: Path,
    asset_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile a fixed CSD into a minimal Gazebo SDF world."""
    blockers = find_csd_realization_blockers(
        csd=csd,
        asset_registry=asset_registry,
        backend=GAZEBO_BACKEND,
    )
    if blockers:
        return CsdCompilationResult(manifest=None, blockers=blockers)

    variants = _variants_by_asset(asset_registry, backend=GAZEBO_BACKEND)
    mesh_blockers = _mesh_path_blockers(
        csd,
        variants,
        Path(asset_root),
        backend=GAZEBO_BACKEND,
    )
    if mesh_blockers:
        return CsdCompilationResult(manifest=None, blockers=mesh_blockers)

    csd_id = _required_str(csd, "csd_id")
    realization_config = dict(realization_config or {})
    variant_hashes = asset_variant_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=GAZEBO_BACKEND,
    )
    cache_key = make_csd_realization_cache_key(
        csd=csd,
        asset_variant_hashes=variant_hashes,
        backend=GAZEBO_BACKEND,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )
    world_root = Path(output_root) / GAZEBO_BACKEND / csd_id
    compiled_asset_root = world_root / "assets"
    diagnostics_root = world_root / "diagnostics"
    world_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(exist_ok=True)
    generated_asset_files = _copy_variant_files(
        csd=csd,
        variants=variants,
        source_asset_root=Path(asset_root),
        compiled_asset_root=compiled_asset_root,
    )
    world_path = world_root / "world.sdf"
    _write_sdf(world_path, csd=csd, variants=variants)
    generated_files = ("manifest.json", "world.sdf", *generated_asset_files)
    manifest = CsdRealizationManifest(
        manifest_id=f"manifest_{GAZEBO_BACKEND}_{csd_id}",
        csd_id=csd_id,
        backend=GAZEBO_BACKEND,
        cache_key=cache_key.digest,
        root_path=str(world_root),
        entry_file="world.sdf",
        generated_files=_unique_files(generated_files),
        preview_files=(),
    )
    _write_manifest(world_root / "manifest.json", manifest)
    return CsdCompilationResult(
        manifest=manifest
    )


def _write_mjcf(
    scene_path: Path,
    *,
    csd: Mapping[str, Any],
    asset_root: Path,
    variants: Mapping[str, Mapping[str, Any]],
    robot_include: str | None = None,
) -> None:
    root = ET.Element("mujoco", {"model": _required_str(csd, "csd_id")})
    if robot_include:
        ET.SubElement(root, "include", {"file": robot_include})
    else:
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
    ET.SubElement(root, "statistic", {"center": "0.3 0 0.4", "extent": "1"})
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
        "camera",
        {
            "name": "world_camera",
            "pos": "1.4 0 1.2",
            "xyaxes": "0 1 0 -0.5 0 0.866",
            "mode": "fixed",
        },
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


def _write_sdf(
    world_path: Path,
    *,
    csd: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> None:
    root = ET.Element("sdf", {"version": "1.12"})
    world = ET.SubElement(root, "world", {"name": _mjcf_name(_required_str(csd, "csd_id"))})
    ET.SubElement(world, "gravity").text = "0 0 -9.81"
    ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    for obj in _csd_objects(csd):
        _append_sdf_model(world, obj, variants)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(world_path, encoding="utf-8", xml_declaration=True)


def _append_sdf_model(
    parent: ET.Element,
    obj: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> None:
    asset_id = _required_str(obj, "asset_id")
    name = _mjcf_name(_required_str(obj, "name"))
    mesh_uri = str(Path("assets") / str(variants[asset_id]["relative_path"]))
    model = ET.SubElement(parent, "model", {"name": name})
    ET.SubElement(model, "pose").text = _sdf_pose_text(obj)
    ET.SubElement(model, "static").text = "true" if bool(obj.get("static", False)) else "false"
    link = ET.SubElement(model, "link", {"name": f"{name}_link"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").text = _object_scalar(obj, "mass_kg", 0.1)
    visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
    _append_sdf_mesh_geometry(visual, mesh_uri)
    collision = ET.SubElement(link, "collision", {"name": f"{name}_collision"})
    _append_sdf_mesh_geometry(collision, mesh_uri)
    surface = ET.SubElement(collision, "surface")
    friction = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction, "ode")
    friction_value = _object_scalar(obj, "friction", 0.7)
    ET.SubElement(ode, "mu").text = friction_value
    ET.SubElement(ode, "mu2").text = friction_value


def _append_sdf_mesh_geometry(parent: ET.Element, mesh_uri: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = mesh_uri


def _copy_variant_files(
    *,
    csd: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    source_asset_root: Path,
    compiled_asset_root: Path,
) -> tuple[str, ...]:
    generated_files: list[str] = []
    for relative_path in _variant_relative_paths(csd, variants):
        source = source_asset_root / relative_path
        destination = compiled_asset_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
        generated_files.append(str(Path("assets") / relative_path))
    return tuple(generated_files)


def _copy_mujoco_robot_template(
    *,
    csd: Mapping[str, Any],
    realization_config: Mapping[str, Any],
    scene_root: Path,
    compiled_asset_root: Path,
) -> tuple[str | None, tuple[str, ...]]:
    template_dir = _mujoco_robot_template_dir(csd, realization_config)
    if template_dir is None:
        return None, ()

    entry_file = str(realization_config.get("robot_template_entry", "panda.xml"))
    source_entry = template_dir / entry_file
    if not source_entry.is_file():
        raise FileNotFoundError(f"MuJoCo robot template entry is missing: {source_entry}")

    robot_dir = scene_root / "assets" / "robots" / template_dir.name
    generated_files: list[str] = []
    for source in template_dir.iterdir():
        if source.is_file() and source.suffix.lower() in {".xml", ".srdf", ".yaml", ".yml"}:
            destination = robot_dir / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
            generated_files.append(str(destination.relative_to(scene_root)))

    mesh_source_root = template_dir / "assets"
    if mesh_source_root.is_dir():
        for source in sorted(path for path in mesh_source_root.rglob("*") if path.is_file()):
            relative_path = source.relative_to(mesh_source_root)
            destination = compiled_asset_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
            generated_files.append(str(destination.relative_to(scene_root)))

    include_path = str((Path("assets") / "robots" / template_dir.name / entry_file).as_posix())
    return include_path, tuple(generated_files)


def _mujoco_robot_template_dir(
    csd: Mapping[str, Any],
    realization_config: Mapping[str, Any],
) -> Path | None:
    configured = realization_config.get("robot_template_dir")
    if configured:
        return Path(str(configured))

    robot_asset_id = str(csd.get("robot_asset_id", ""))
    if robot_asset_id in {"robot_franka_panda", "franka_panda"}:
        return (
            Path(__file__).resolve().parents[2]
            / "drivers_sim"
            / "mujoco"
            / "assets"
            / "robots"
            / "franka_panda"
        )
    return None


def _mujoco_robot_template_blockers(
    *,
    csd: Mapping[str, Any],
    realization_config: Mapping[str, Any],
) -> tuple[CsdRealizationBlocker, ...]:
    robot_asset_id = str(csd.get("robot_asset_id", ""))
    if not robot_asset_id:
        return ()

    csd_id = _required_str(csd, "csd_id")
    template_dir = _mujoco_robot_template_dir(csd, realization_config)
    if template_dir is None:
        return (
            _asset_blocker(
                csd_id,
                robot_asset_id,
                MUJOCO_BACKEND,
                "no MuJoCo robot template is configured for robot asset",
            ),
        )
    if not template_dir.is_dir():
        return (
            _asset_blocker(
                csd_id,
                robot_asset_id,
                MUJOCO_BACKEND,
                f"MuJoCo robot template directory is missing: {template_dir}",
            ),
        )
    entry_file = str(realization_config.get("robot_template_entry", "panda.xml"))
    if not (template_dir / entry_file).is_file():
        return (
            _asset_blocker(
                csd_id,
                robot_asset_id,
                MUJOCO_BACKEND,
                f"MuJoCo robot template entry is missing: {entry_file}",
            ),
        )
    return ()


def _write_manifest(path: Path, manifest: CsdRealizationManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unique_files(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(paths))


def _variant_relative_paths(
    csd: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> Iterable[Path]:
    for asset_id in _csd_asset_ids(csd):
        yield Path(str(variants[asset_id]["relative_path"]))


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
    *,
    backend: str,
) -> tuple[CsdRealizationBlocker, ...]:
    csd_id = _required_str(csd, "csd_id")
    blockers: list[CsdRealizationBlocker] = []
    for asset_id in _csd_asset_ids(csd):
        relative_path = str(variants[asset_id].get("relative_path", ""))
        if not relative_path:
            blockers.append(
                _asset_blocker(csd_id, asset_id, backend, "asset variant has no relative_path")
            )
            continue
        if Path(relative_path).is_absolute():
            blockers.append(
                _asset_blocker(csd_id, asset_id, backend, "asset variant path must be relative")
            )
            continue
        if not (asset_root / relative_path).is_file():
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"asset variant file is missing: {relative_path}",
                )
            )
    return tuple(blockers)


def _asset_blocker(
    csd_id: str,
    asset_id: str,
    backend: str,
    reason: str,
) -> CsdRealizationBlocker:
    return CsdRealizationBlocker(
        blocker_id=f"{csd_id}_{backend}_{asset_id}_compile_blocked",
        csd_id=csd_id,
        backend=backend,
        asset_id=asset_id,
        scope="asset",
        reason=reason,
    )


def _variants_by_asset(
    asset_registry: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, Mapping[str, Any]]:
    variants: dict[str, Mapping[str, Any]] = {}
    for record in (*asset_registry.get("objects", ()), *asset_registry.get("assets", ())):
        if not isinstance(record, Mapping):
            continue
        asset_id = str(record.get("object_id") or record.get("asset_id") or "")
        if asset_id and asset_id not in variants:
            variant = _passed_variant(record.get("variants", ()), backend=backend)
            if variant is not None:
                variants[asset_id] = variant
    return variants


def _passed_variant(
    variant_records: object,
    *,
    backend: str,
) -> Mapping[str, Any] | None:
    if not isinstance(variant_records, (list, tuple)):
        return None
    for variant in variant_records:
        if not isinstance(variant, Mapping):
            continue
        if (
            str(variant.get("engine", "")) == backend
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


def _sdf_pose_text(obj: Mapping[str, Any]) -> str:
    position = _pose_part(obj, "position")
    orientation = _pose_part(obj, "orientation")
    roll, pitch, yaw = _quaternion_to_rpy(
        w=float(orientation["w"]),
        x=float(orientation["x"]),
        y=float(orientation["y"]),
        z=float(orientation["z"]),
    )
    return _numbers_text((position["x"], position["y"], position["z"], roll, pitch, yaw))


def _quaternion_to_rpy(*, w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _pose_part(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    pose = obj.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("CSD object pose is required for compilation")
    part = pose.get(key)
    if not isinstance(part, Mapping):
        raise ValueError(f"CSD object pose.{key} is required for compilation")
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
