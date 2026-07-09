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
    BackendResourceAdapter,
    BackendResourceMaterial,
    ConcreteScenarioDefinition,
    CsdObject,
    CsdRealizationBlocker,
    CsdRealizationManifest,
    asset_resource_hashes_for_csd,
    backend_resource_adapters_by_asset,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)

MUJOCO_BACKEND = "mujoco"
GAZEBO_BACKEND = "gazebo"
DEFAULT_REALIZATION_VERSION = "csd-compiler-0.2"
DEFAULT_MUJOCO_GRAVITY = (0.0, 0.0, -9.81)
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

    typed_csd = ConcreteScenarioDefinition.from_mapping(csd)
    realization_config = dict(realization_config or {})

    resources = backend_resource_adapters_by_asset(asset_registry, backend=MUJOCO_BACKEND)
    mesh_blockers = _mesh_path_blockers(
        csd,
        resources,
        Path(asset_root),
        backend=MUJOCO_BACKEND,
    )
    if mesh_blockers:
        return CsdCompilationResult(manifest=None, blockers=mesh_blockers)

    csd_id = _required_str(csd, "csd_id")
    robot_blockers = _mujoco_robot_template_blockers(
        csd=csd,
        realization_config=realization_config,
    )
    if robot_blockers:
        return CsdCompilationResult(manifest=None, blockers=robot_blockers)

    semantic_blockers = _mujoco_csd_semantic_blockers(
        typed_csd,
        uses_robot_template=_mujoco_robot_template_dir(csd, realization_config) is not None,
    )
    if semantic_blockers:
        return CsdCompilationResult(manifest=None, blockers=semantic_blockers)

    resource_hashes = asset_resource_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=MUJOCO_BACKEND,
    )
    cache_key = make_csd_realization_cache_key(
        csd=csd,
        asset_variant_hashes=resource_hashes,
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
    generated_asset_files = _copy_resource_files(
        csd=csd,
        resources=resources,
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
        csd=typed_csd,
        asset_root=compiled_asset_root,
        resources=resources,
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

    resources = backend_resource_adapters_by_asset(asset_registry, backend=GAZEBO_BACKEND)
    mesh_blockers = _mesh_path_blockers(
        csd,
        resources,
        Path(asset_root),
        backend=GAZEBO_BACKEND,
    )
    if mesh_blockers:
        return CsdCompilationResult(manifest=None, blockers=mesh_blockers)

    csd_id = _required_str(csd, "csd_id")
    realization_config = dict(realization_config or {})
    resource_hashes = asset_resource_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=GAZEBO_BACKEND,
    )
    cache_key = make_csd_realization_cache_key(
        csd=csd,
        asset_variant_hashes=resource_hashes,
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
    generated_asset_files = _copy_resource_files(
        csd=csd,
        resources=resources,
        source_asset_root=Path(asset_root),
        compiled_asset_root=compiled_asset_root,
    )
    world_path = world_root / "world.sdf"
    _write_sdf(world_path, csd=csd, resources=resources)
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
    csd: ConcreteScenarioDefinition,
    asset_root: Path,
    resources: Mapping[str, BackendResourceAdapter],
    robot_include: str | None = None,
) -> None:
    root = ET.Element("mujoco", {"model": csd.csd_id})
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
                "texturedir": str(asset_root),
            },
        )
        ET.SubElement(root, "option", {"gravity": _vector3_text(csd.environment.gravity)})
    ET.SubElement(root, "statistic", {"center": "0.3 0 0.4", "extent": "1"})
    assets = ET.SubElement(root, "asset")
    for asset_id in _csd_asset_ids(csd.raw):
        resource = resources[asset_id]
        mesh_attrs = {
            "name": _mjcf_name(asset_id),
            "file": resource.mesh_path,
        }
        mesh_scale = _mesh_scale_text(resource)
        if mesh_scale is not None:
            mesh_attrs["scale"] = mesh_scale
        ET.SubElement(assets, "mesh", mesh_attrs)
        if resource.collision_mesh_path:
            ET.SubElement(
                assets,
                "mesh",
                {
                    "name": f"{_mjcf_name(asset_id)}_collision",
                    "file": resource.collision_mesh_path,
                },
            )
        material = resource.material
        if material is not None:
            _append_mjcf_material(assets, material)

    worldbody = ET.SubElement(root, "worldbody")
    _append_lights(worldbody, csd)
    _append_cameras(worldbody, csd)
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
    _append_environment_surfaces(worldbody, csd)
    for obj in csd.objects:
        _append_object_body(worldbody, obj, resources)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)


def _write_sdf(
    world_path: Path,
    *,
    csd: Mapping[str, Any],
    resources: Mapping[str, BackendResourceAdapter],
) -> None:
    root = ET.Element("sdf", {"version": "1.12"})
    world = ET.SubElement(root, "world", {"name": _mjcf_name(_required_str(csd, "csd_id"))})
    ET.SubElement(world, "gravity").text = "0 0 -9.81"
    ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    for obj in _csd_objects(csd):
        _append_sdf_model(world, obj, resources)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(world_path, encoding="utf-8", xml_declaration=True)


def _append_sdf_model(
    parent: ET.Element,
    obj: Mapping[str, Any],
    resources: Mapping[str, BackendResourceAdapter],
) -> None:
    asset_id = _required_str(obj, "asset_id")
    name = _mjcf_name(_required_str(obj, "name"))
    mesh_uri = str(Path("assets") / resources[asset_id].mesh_path)
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


def _append_lights(parent: ET.Element, csd: ConcreteScenarioDefinition) -> None:
    if not csd.environment.lighting:
        ET.SubElement(
            parent,
            "light",
            {"name": "key_light", "pos": "0 -1 3", "dir": "0 0 -1"},
        )
        return
    for light in csd.environment.lighting:
        attrs = {
            "name": _mjcf_name(light.light_id),
            "pos": _vector3_text(light.position),
            "dir": _vector3_text(light.direction),
        }
        ET.SubElement(parent, "light", attrs)


def _append_cameras(parent: ET.Element, csd: ConcreteScenarioDefinition) -> None:
    if not csd.environment.cameras:
        ET.SubElement(
            parent,
            "camera",
            {
                "name": "world_camera",
                "pos": "1.4 0 1.2",
                "xyaxes": "0 1 0 -0.5 0 0.866",
                "mode": "fixed",
            },
        )
        return
    for camera in csd.environment.cameras:
        attrs = {
            "name": _mjcf_name(camera.camera_id),
            "pos": _vector3_text(camera.position),
            "mode": camera.mode,
        }
        if camera.xyaxes is not None:
            attrs["xyaxes"] = _numbers_text(camera.xyaxes)
        ET.SubElement(parent, "camera", attrs)


def _append_environment_surfaces(parent: ET.Element, csd: ConcreteScenarioDefinition) -> None:
    for surface in csd.environment.surfaces:
        if surface.surface_type != "box":
            continue
        surface_id = _mjcf_name(surface.surface_id)
        body = ET.SubElement(
            parent,
            "body",
            {
                "name": surface_id,
                "pos": _vector3_text(surface.pose.position),
                "quat": _quaternion_text(surface.pose.orientation),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{surface_id}_geom",
                "type": "box",
                "size": _vector3_text(surface.size),
                "rgba": _numbers_text(surface.rgba),
                "friction": _numbers_text(surface.friction),
            },
        )


def _copy_resource_files(
    *,
    csd: Mapping[str, Any],
    resources: Mapping[str, BackendResourceAdapter],
    source_asset_root: Path,
    compiled_asset_root: Path,
) -> tuple[str, ...]:
    generated_files: list[str] = []
    for relative_path in _resource_relative_paths(csd, resources):
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

    robot_asset_id = _robot_asset_id(csd)
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
    robot_asset_id = _robot_asset_id(csd)
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


def _mujoco_csd_semantic_blockers(
    csd: ConcreteScenarioDefinition,
    *,
    uses_robot_template: bool,
) -> tuple[CsdRealizationBlocker, ...]:
    blockers: list[CsdRealizationBlocker] = []
    if csd.units != "m":
        blockers.append(
            _csd_blocker(
                csd.csd_id,
                "scenario_units",
                f"MuJoCo compiler supports only CSD units='m', got '{csd.units}'",
            )
        )
    if csd.frame != "world":
        blockers.append(
            _csd_blocker(
                csd.csd_id,
                "scenario_frame",
                f"MuJoCo compiler supports only frame='world', got '{csd.frame}'",
            )
        )
    for surface in csd.environment.surfaces:
        if surface.surface_type != "box":
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    surface.surface_id,
                    (
                        "MuJoCo compiler does not support environment surface "
                        f"type '{surface.surface_type}'"
                    ),
                )
            )
    if uses_robot_template and not _is_default_mujoco_gravity(csd):
        robot_asset_id = csd.robot.asset_id if csd.robot is not None else "robot_template"
        blockers.append(
            _csd_blocker(
                csd.csd_id,
                robot_asset_id,
                (
                    "MuJoCo robot template gravity override is not implemented; "
                    "template scenes require default gravity 0 0 -9.81"
                ),
            )
        )
    return tuple(blockers)


def _is_default_mujoco_gravity(csd: ConcreteScenarioDefinition) -> bool:
    gravity = csd.environment.gravity
    return (
        math.isclose(gravity.x, DEFAULT_MUJOCO_GRAVITY[0])
        and math.isclose(gravity.y, DEFAULT_MUJOCO_GRAVITY[1])
        and math.isclose(gravity.z, DEFAULT_MUJOCO_GRAVITY[2])
    )


def _write_manifest(path: Path, manifest: CsdRealizationManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unique_files(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(paths))


def _resource_relative_paths(
    csd: Mapping[str, Any],
    resources: Mapping[str, BackendResourceAdapter],
) -> Iterable[Path]:
    for asset_id in _csd_asset_ids(csd):
        resource = resources[asset_id]
        yield Path(resource.mesh_path)
        if resource.collision_mesh_path:
            yield Path(resource.collision_mesh_path)
        if resource.material is not None and resource.material.texture_path:
            yield Path(resource.material.texture_path)


def _append_object_body(
    parent: ET.Element,
    obj: CsdObject,
    resources: Mapping[str, BackendResourceAdapter],
) -> None:
    asset_id = obj.asset_id
    resource = resources[asset_id]
    body = ET.SubElement(
        parent,
        "body",
        {
            "name": _mjcf_name(obj.name),
            "pos": _vector3_text(obj.pose.position),
            "quat": _quaternion_text(obj.pose.orientation),
        },
    )
    if not obj.static:
        ET.SubElement(body, "freejoint")
    visual_geom_attrs = {
        "name": f"{_mjcf_name(obj.name)}_geom",
        "type": "mesh",
        "mesh": _mjcf_name(asset_id),
        "rgba": "0.7 0.7 0.7 1",
    }
    material = resource.material
    if material is not None:
        visual_geom_attrs["material"] = _mjcf_name(material.name or f"{asset_id}_material")
        visual_geom_attrs.pop("rgba")
    if resource.collision_mesh_path:
        visual_geom_attrs["contype"] = "0"
        visual_geom_attrs["conaffinity"] = "0"
        ET.SubElement(body, "geom", visual_geom_attrs)
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{_mjcf_name(obj.name)}_collision_geom",
                "type": "mesh",
                "mesh": f"{_mjcf_name(asset_id)}_collision",
                "mass": _number_text(obj.initial_state.mass_kg),
                "friction": _numbers_text(obj.initial_state.friction),
                "rgba": "0 0 0 0",
            },
        )
        return

    visual_geom_attrs["mass"] = _number_text(obj.initial_state.mass_kg)
    visual_geom_attrs["friction"] = _numbers_text(obj.initial_state.friction)
    ET.SubElement(body, "geom", visual_geom_attrs)


def _mesh_path_blockers(
    csd: Mapping[str, Any],
    resources: Mapping[str, BackendResourceAdapter],
    asset_root: Path,
    *,
    backend: str,
) -> tuple[CsdRealizationBlocker, ...]:
    csd_id = _required_str(csd, "csd_id")
    blockers: list[CsdRealizationBlocker] = []
    for asset_id in _csd_asset_ids(csd):
        relative_path = resources[asset_id].mesh_path
        if not relative_path:
            blockers.append(
                _asset_blocker(csd_id, asset_id, backend, "backend resource has no mesh_path")
            )
            continue
        if Path(relative_path).is_absolute():
            blockers.append(
                _asset_blocker(csd_id, asset_id, backend, "backend resource path must be relative")
            )
            continue
        if not (asset_root / relative_path).is_file():
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"backend resource mesh file is missing: {relative_path}",
                )
            )
            continue
        collision_mesh_path = resources[asset_id].collision_mesh_path
        if collision_mesh_path:
            if Path(collision_mesh_path).is_absolute():
                blockers.append(
                    _asset_blocker(
                        csd_id,
                        asset_id,
                        backend,
                        "asset collision mesh path must be relative",
                    )
                )
            elif not (asset_root / collision_mesh_path).is_file():
                blockers.append(
                    _asset_blocker(
                        csd_id,
                        asset_id,
                        backend,
                        f"asset collision mesh file is missing: {collision_mesh_path}",
                    )
                )
        material = resources[asset_id].material
        if material is None or not material.texture_path:
            continue
        texture_path = material.texture_path
        if Path(texture_path).is_absolute():
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    "asset material texture path must be relative",
                )
            )
            continue
        if not (asset_root / texture_path).is_file():
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"asset material texture file is missing: {texture_path}",
                )
            )
    return tuple(blockers)


def _append_mjcf_material(parent: ET.Element, material: BackendResourceMaterial) -> None:
    material_name = _mjcf_name(material.name or "material")
    material_attrs = {"name": material_name}
    if material.texture_path:
        texture_name = f"{material_name}_texture"
        ET.SubElement(
            parent,
            "texture",
            {
                "name": texture_name,
                "type": "2d",
                "file": material.texture_path,
            },
        )
        material_attrs["texture"] = texture_name
    if material.rgba is not None:
        material_attrs["rgba"] = _numbers_text(material.rgba)
    ET.SubElement(parent, "material", material_attrs)


def _mesh_scale_text(resource: BackendResourceAdapter) -> str | None:
    scale = resource.mesh_scale
    if scale is None:
        return None
    if isinstance(scale, (int, float)):
        return _numbers_text((scale, scale, scale))
    return _numbers_text(scale)


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


def _csd_blocker(
    csd_id: str,
    subject_id: str,
    reason: str,
) -> CsdRealizationBlocker:
    return CsdRealizationBlocker(
        blocker_id=f"{csd_id}_{MUJOCO_BACKEND}_{subject_id}_compile_blocked",
        csd_id=csd_id,
        backend=MUJOCO_BACKEND,
        asset_id=subject_id,
        scope="csd",
        reason=reason,
    )


def _csd_objects(csd: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(obj for obj in _csd_scenario(csd).get("objects", ()) if isinstance(obj, Mapping))


def _csd_asset_ids(csd: Mapping[str, Any]) -> tuple[str, ...]:
    asset_ids = [str(obj["asset_id"]) for obj in _csd_objects(csd) if obj.get("asset_id")]
    return tuple(dict.fromkeys(asset_ids))


def _csd_scenario(csd: Mapping[str, Any]) -> Mapping[str, Any]:
    scenario = csd.get("scenario")
    return scenario if isinstance(scenario, Mapping) else csd


def _robot_asset_id(csd: Mapping[str, Any]) -> str:
    robot = _csd_scenario(csd).get("robot", {})
    if isinstance(robot, Mapping):
        return str(robot.get("asset_id", ""))
    return str(csd.get("robot_asset_id", ""))


def _vector3_text(vector: Any) -> str:
    return _numbers_text((vector.x, vector.y, vector.z))


def _quaternion_text(quaternion: Any) -> str:
    return _numbers_text((quaternion.w, quaternion.x, quaternion.y, quaternion.z))


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


def _object_scalar(obj: Mapping[str, Any] | CsdObject, key: str, default: float) -> str:
    initial_state = (
        obj.initial_state
        if isinstance(obj, CsdObject)
        else obj.get("initial_state", {})
    )
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
