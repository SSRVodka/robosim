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
    CsdRealizationValidationRecord,
    CsdRelationship,
    CsdRelationshipType,
    CsdSurface,
    asset_resource_hashes_for_csd,
    backend_resource_adapters_by_asset,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)

MUJOCO_BACKEND = "mujoco"
GAZEBO_BACKEND = "gazebo"
DEFAULT_REALIZATION_VERSION = "csd-compiler-0.3"
MUJOCO_MESH_EXTENSIONS = frozenset({".obj", ".stl", ".msh"})
MUJOCO_PREVIEW_SIZE_PX = 512


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
    csd_id = _required_str(csd, "csd_id")
    simulator_version = simulator_version or _mujoco_simulator_version()

    semantic_blockers = _mujoco_csd_semantic_blockers(
        typed_csd
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
    cached_manifest = _cached_manifest(scene_root, cache_key.digest)
    if cached_manifest is not None:
        return CsdCompilationResult(manifest=cached_manifest)

    resources = backend_resource_adapters_by_asset(asset_registry, backend=MUJOCO_BACKEND)
    mesh_blockers = _mesh_path_blockers(
        csd,
        resources,
        Path(asset_root),
        backend=MUJOCO_BACKEND,
    )
    if mesh_blockers:
        return CsdCompilationResult(manifest=None, blockers=mesh_blockers)

    robot_blockers = _mujoco_robot_template_blockers(
        csd=csd,
        realization_config=realization_config,
    )
    if robot_blockers:
        return CsdCompilationResult(manifest=None, blockers=robot_blockers)

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
        csd=typed_csd,
        raw_csd=csd,
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
    load_check_file, load_check_blockers = _write_mujoco_load_check(
        scene_path=scene_path,
        diagnostics_root=diagnostics_root,
        csd=typed_csd,
    )
    if load_check_blockers:
        return CsdCompilationResult(manifest=None, blockers=load_check_blockers)
    relationship_check_file, relationship_check_blockers = _write_mujoco_relationship_check(
        scene_path=scene_path,
        diagnostics_root=diagnostics_root,
        csd=typed_csd,
    )
    if relationship_check_blockers:
        return CsdCompilationResult(manifest=None, blockers=relationship_check_blockers)
    physics_check_file, physics_check_blockers = _write_mujoco_physics_check(
        scene_path=scene_path,
        diagnostics_root=diagnostics_root,
        csd=typed_csd,
    )
    if physics_check_blockers:
        return CsdCompilationResult(manifest=None, blockers=physics_check_blockers)
    preview_file, preview_blockers = _write_mujoco_preview(
        scene_path=scene_path,
        diagnostics_root=diagnostics_root,
        csd=typed_csd,
    )
    if preview_blockers:
        return CsdCompilationResult(manifest=None, blockers=preview_blockers)
    validation_record_file = "diagnostics/validation_record.json"
    generated_files = (
        "manifest.json",
        "scene.xml",
        load_check_file,
        relationship_check_file,
        physics_check_file,
        validation_record_file,
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
        preview_files=(preview_file,),
    )
    _write_validation_record(
        scene_root / validation_record_file,
        CsdRealizationValidationRecord(
            validation_id=f"validation_{MUJOCO_BACKEND}_{csd_id}",
            csd_id=csd_id,
            backend=MUJOCO_BACKEND,
            manifest_id=manifest.manifest_id,
            cache_key=manifest.cache_key,
            status="passed",
            evidence_files=(load_check_file, relationship_check_file, physics_check_file),
            preview_files=manifest.preview_files,
        ),
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
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "offwidth": str(MUJOCO_PREVIEW_SIZE_PX),
            "offheight": str(MUJOCO_PREVIEW_SIZE_PX),
        },
    )
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
    csd: ConcreteScenarioDefinition,
    raw_csd: Mapping[str, Any],
    realization_config: Mapping[str, Any],
    scene_root: Path,
    compiled_asset_root: Path,
) -> tuple[str | None, tuple[str, ...]]:
    template_dir = _mujoco_robot_template_dir(raw_csd, realization_config)
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

    entry_destination = robot_dir / entry_file
    _patch_mujoco_template_gravity(
        entry_destination,
        gravity=_vector3_text(csd.environment.gravity),
    )

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
        if not _positive_finite_vector(surface.size):
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    surface.surface_id,
                    f"surface {surface.surface_id} box size values must be positive and finite",
                )
            )
        if _quaternion_norm(surface.pose.orientation) == 0.0:
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    surface.surface_id,
                    f"surface {surface.surface_id} orientation quaternion must be non-zero",
                )
            )
    for camera in csd.environment.cameras:
        if camera.xyaxes is not None and not _valid_camera_xyaxes(camera.xyaxes):
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    camera.camera_id,
                    (
                        f"camera {camera.camera_id} xyaxes must contain non-zero "
                        "non-parallel axes"
                    ),
                )
            )
    for light in csd.environment.lighting:
        if _vector_norm((light.direction.x, light.direction.y, light.direction.z)) == 0.0:
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    light.light_id,
                    f"light {light.light_id} direction must be non-zero",
                )
            )
    for obj in csd.objects:
        if _quaternion_norm(obj.pose.orientation) == 0.0:
            blockers.append(
                _csd_blocker(
                    csd.csd_id,
                    obj.name,
                    f"object {obj.name} orientation quaternion must be non-zero",
                )
            )
        blockers.extend(_mujoco_object_physical_blockers(csd.csd_id, obj))
    known_entities = _csd_entity_refs(csd)
    for relationship in csd.relationships:
        for field_name, entity_ref in (
            ("subject", relationship.subject),
            ("object", relationship.object),
        ):
            if entity_ref not in known_entities:
                blockers.append(
                    _csd_blocker(
                        csd.csd_id,
                        relationship.relation_id,
                        (
                            f"relationship {relationship.relation_id} {field_name} "
                            f"references unknown entity: {entity_ref}"
                        ),
                    )
                )
    return tuple(blockers)


def _csd_entity_refs(csd: ConcreteScenarioDefinition) -> set[str]:
    refs = {f"object:{obj.name}" for obj in csd.objects}
    refs.update(f"surface:{surface.surface_id}" for surface in csd.environment.surfaces)
    if csd.robot is not None:
        refs.add(f"robot:{csd.robot.asset_id}")
    return refs


def _patch_mujoco_template_gravity(path: Path, *, gravity: str) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        root.insert(1 if compiler is not None else 0, option)
    option.set("gravity", gravity)
    ET.indent(root, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)


def _write_manifest(path: Path, manifest: CsdRealizationManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_validation_record(
    path: Path,
    record: CsdRealizationValidationRecord,
) -> None:
    path.write_text(
        json.dumps(record.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mujoco_simulator_version() -> str | None:
    try:
        import mujoco
    except Exception:
        return None
    return str(getattr(mujoco, "__version__", "")) or None


def _cached_manifest(scene_root: Path, cache_key: str) -> CsdRealizationManifest | None:
    manifest_path = scene_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = CsdRealizationManifest.from_json_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.cache_key != cache_key or manifest.backend != MUJOCO_BACKEND:
        return None
    if not _manifest_files_exist(scene_root, manifest):
        return None
    return manifest


def _manifest_files_exist(scene_root: Path, manifest: CsdRealizationManifest) -> bool:
    for relative_path in (*manifest.generated_files, *manifest.preview_files):
        if not (scene_root / relative_path).is_file():
            return False
    return True


def _write_mujoco_load_check(
    *,
    scene_path: Path,
    diagnostics_root: Path,
    csd: ConcreteScenarioDefinition,
) -> tuple[str, tuple[CsdRealizationBlocker, ...]]:
    checks: list[dict[str, object]] = []
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        checks.append(
            _load_check(
                "model_load",
                passed=True,
                details={"nbody": int(model.nbody), "ngeom": int(model.ngeom)},
            )
        )
        checks.append(
            _load_check(
                "gravity",
                expected=_vector3_json(csd.environment.gravity),
                actual=_float_sequence_json(model.opt.gravity),
            )
        )
        for camera in csd.environment.cameras:
            camera_name = _mjcf_name(camera.camera_id)
            try:
                loaded_camera = model.camera(camera_name)
                checks.append(
                    _load_check(
                        f"camera_pose:{camera_name}",
                        expected=_vector3_json(camera.position),
                        actual=_float_sequence_json(loaded_camera.pos),
                    )
                )
                if camera.xyaxes is not None:
                    checks.append(
                        _load_check(
                            f"camera_orientation:{camera_name}",
                            expected=_camera_xyaxes_quaternion_json(camera.xyaxes),
                            actual=_quaternion_sequence_json(loaded_camera.quat),
                        )
                    )
            except KeyError:
                checks.append(
                    _load_check(
                        f"camera_pose:{camera_name}",
                        passed=False,
                        expected=_vector3_json(camera.position),
                        details={"reason": "camera not found in loaded MuJoCo model"},
                    )
                )
                if camera.xyaxes is not None:
                    checks.append(
                        _load_check(
                            f"camera_orientation:{camera_name}",
                            passed=False,
                            expected=_camera_xyaxes_quaternion_json(camera.xyaxes),
                            details={"reason": "camera not found in loaded MuJoCo model"},
                        )
                    )
        for light in csd.environment.lighting:
            light_name = _mjcf_name(light.light_id)
            try:
                loaded_light = model.light(light_name)
                checks.append(
                    _load_check(
                        f"light_pose:{light_name}",
                        expected=_vector3_json(light.position),
                        actual=_float_sequence_json(loaded_light.pos),
                    )
                )
                checks.append(
                    _load_check(
                        f"light_direction:{light_name}",
                        expected=_vector3_json(light.direction),
                        actual=_float_sequence_json(loaded_light.dir),
                    )
                )
            except KeyError:
                checks.append(
                    _load_check(
                        f"light_pose:{light_name}",
                        passed=False,
                        expected=_vector3_json(light.position),
                        details={"reason": "light not found in loaded MuJoCo model"},
                    )
                )
                checks.append(
                    _load_check(
                        f"light_direction:{light_name}",
                        passed=False,
                        expected=_vector3_json(light.direction),
                        details={"reason": "light not found in loaded MuJoCo model"},
                    )
                )
        for obj in csd.objects:
            body_name = _mjcf_name(obj.name)
            try:
                body = model.body(body_name)
                actual = _float_sequence_json(body.pos)
                expected = _vector3_json(obj.pose.position)
                checks.append(
                    _load_check(
                        f"body_pose:{body_name}",
                        expected=expected,
                        actual=actual,
                    )
                )
                checks.append(
                    _load_check(
                        f"body_orientation:{body_name}",
                        expected=_quaternion_json(obj.pose.orientation),
                        actual=_quaternion_sequence_json(body.quat),
                    )
                )
                checks.append(
                    _load_check(
                        f"body_mass:{body_name}",
                        expected=[_json_float(obj.initial_state.mass_kg)],
                        actual=_float_sequence_json(body.mass),
                    )
                )
                if obj.initial_state.inertial is not None:
                    checks.append(
                        _load_check(
                            f"body_inertial_pos:{body_name}",
                            expected=_vector3_json(
                                obj.initial_state.inertial.center_of_mass
                            ),
                            actual=_float_sequence_json(body.ipos),
                        )
                    )
                    checks.append(
                        _load_check(
                            f"body_inertia:{body_name}",
                            expected=_float_sequence_json(
                                obj.initial_state.inertial.diagonal_inertia_kg_m2
                            ),
                            actual=_float_sequence_json(body.inertia),
                        )
                    )
            except KeyError:
                checks.append(
                    _load_check(
                        f"body_pose:{body_name}",
                        passed=False,
                        expected=_vector3_json(obj.pose.position),
                        details={"reason": "body not found in loaded MuJoCo model"},
                    )
                )
                checks.append(
                    _load_check(
                        f"body_orientation:{body_name}",
                        passed=False,
                        expected=_quaternion_json(obj.pose.orientation),
                        details={"reason": "body not found in loaded MuJoCo model"},
                    )
                )
                checks.append(
                    _load_check(
                        f"body_mass:{body_name}",
                        passed=False,
                        expected=[_json_float(obj.initial_state.mass_kg)],
                        details={"reason": "body not found in loaded MuJoCo model"},
                    )
                )
                if obj.initial_state.inertial is not None:
                    checks.append(
                        _load_check(
                            f"body_inertial_pos:{body_name}",
                            passed=False,
                            expected=_vector3_json(
                                obj.initial_state.inertial.center_of_mass
                            ),
                            details={"reason": "body not found in loaded MuJoCo model"},
                        )
                    )
                    checks.append(
                        _load_check(
                            f"body_inertia:{body_name}",
                            passed=False,
                            expected=_float_sequence_json(
                                obj.initial_state.inertial.diagonal_inertia_kg_m2
                            ),
                            details={"reason": "body not found in loaded MuJoCo model"},
                        )
                    )
            geom_name = _mujoco_collision_geom_name(model, body_name)
            try:
                geom = model.geom(geom_name)
                checks.append(
                    _load_check(
                        f"geom_friction:{geom_name}",
                        expected=_float_sequence_json(obj.initial_state.friction),
                        actual=_float_sequence_json(geom.friction),
                    )
                )
                if obj.initial_state.contact is not None:
                    checks.append(
                        _load_check(
                            f"geom_contact:{geom_name}",
                            expected=_mujoco_contact_check_values(obj),
                            actual=_mujoco_contact_check_values_from_geom(geom, obj),
                        )
                    )
            except KeyError:
                checks.append(
                    _load_check(
                        f"geom_friction:{geom_name}",
                        passed=False,
                        expected=_float_sequence_json(obj.initial_state.friction),
                        details={"reason": "collision-bearing geom not found"},
                    )
                )
        for surface in csd.environment.surfaces:
            body_name = _mjcf_name(surface.surface_id)
            try:
                body = model.body(body_name)
                actual = _float_sequence_json(body.pos)
                expected = _vector3_json(surface.pose.position)
                checks.append(
                    _load_check(
                        f"surface_pose:{body_name}",
                        expected=expected,
                        actual=actual,
                    )
                )
                checks.append(
                    _load_check(
                        f"surface_orientation:{body_name}",
                        expected=_quaternion_json(surface.pose.orientation),
                        actual=_quaternion_sequence_json(body.quat),
                    )
                )
                geom_name = f"{body_name}_geom"
                try:
                    geom = model.geom(geom_name)
                    checks.append(
                        _load_check(
                            f"surface_size:{geom_name}",
                            expected=_vector3_json(surface.size),
                            actual=_float_sequence_json(geom.size),
                        )
                    )
                    checks.append(
                        _load_check(
                            f"surface_friction:{geom_name}",
                            expected=_float_sequence_json(surface.friction),
                            actual=_float_sequence_json(geom.friction),
                        )
                    )
                    checks.append(
                        _load_check(
                            f"surface_rgba:{geom_name}",
                            expected=_rgba_sequence_json(surface.rgba),
                            actual=_rgba_sequence_json(geom.rgba),
                        )
                    )
                except KeyError:
                    checks.append(
                        _load_check(
                            f"surface_size:{geom_name}",
                            passed=False,
                            expected=_vector3_json(surface.size),
                            details={"reason": "surface geom not found in loaded MuJoCo model"},
                        )
                    )
            except KeyError:
                checks.append(
                    _load_check(
                        f"surface_pose:{body_name}",
                        passed=False,
                        expected=_vector3_json(surface.pose.position),
                        details={"reason": "surface body not found in loaded MuJoCo model"},
                    )
                )
                checks.append(
                    _load_check(
                        f"surface_orientation:{body_name}",
                        passed=False,
                        expected=_quaternion_json(surface.pose.orientation),
                        details={"reason": "surface body not found in loaded MuJoCo model"},
                    )
                )
    except Exception as exc:
        checks.append(
            _load_check(
                "model_load",
                passed=False,
                details={"reason": str(exc)},
            )
        )

    passed = all(check["status"] == "passed" for check in checks)
    payload = {
        "schema_version": "0.1",
        "backend": MUJOCO_BACKEND,
        "csd_id": csd.csd_id,
        "entry_file": scene_path.name,
        "status": "passed" if passed else "failed",
        "checks": checks,
    }
    path = diagnostics_root / "load_check.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if passed:
        return "diagnostics/load_check.json", ()
    return (
        "diagnostics/load_check.json",
        (
            CsdRealizationBlocker(
                blocker_id=f"{csd.csd_id}_{MUJOCO_BACKEND}_scene_load_check_failed",
                csd_id=csd.csd_id,
                backend=MUJOCO_BACKEND,
                asset_id="scene",
                scope="vsim_realization",
                reason="generated MuJoCo scene failed compiler load-check diagnostics",
            ),
        ),
    )


def _load_check(
    name: str,
    *,
    passed: bool | None = None,
    expected: object | None = None,
    actual: object | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if passed is None:
        passed = expected == actual
    check: dict[str, object] = {
        "name": name,
        "status": "passed" if passed else "failed",
    }
    if expected is not None:
        check["expected"] = expected
    if actual is not None:
        check["actual"] = actual
    if details is not None:
        check["details"] = dict(details)
    return check


def _mujoco_collision_geom_name(model: Any, body_name: str) -> str:
    collision_geom_name = f"{body_name}_collision_geom"
    try:
        model.geom(collision_geom_name)
        return collision_geom_name
    except KeyError:
        return f"{body_name}_geom"


def _mujoco_contact_check_values(obj: CsdObject) -> dict[str, list[float]]:
    contact = obj.initial_state.contact
    if contact is None:
        return {}
    values: dict[str, list[float]] = {}
    if contact.margin_m is not None:
        values["margin"] = [_json_float(contact.margin_m)]
    if contact.gap_m is not None:
        values["gap"] = [_json_float(contact.gap_m)]
    if contact.solref is not None:
        values["solref"] = _float_sequence_json(contact.solref)
    if contact.solimp is not None:
        values["solimp"] = _float_sequence_json(contact.solimp)
    return values


def _mujoco_contact_check_values_from_geom(
    geom: Any,
    obj: CsdObject,
) -> dict[str, list[float]]:
    contact = obj.initial_state.contact
    if contact is None:
        return {}
    values: dict[str, list[float]] = {}
    if contact.margin_m is not None:
        values["margin"] = _float_sequence_json(geom.margin)
    if contact.gap_m is not None:
        values["gap"] = _float_sequence_json(geom.gap)
    if contact.solref is not None:
        values["solref"] = _float_sequence_json(geom.solref)
    if contact.solimp is not None:
        values["solimp"] = _float_sequence_json(geom.solimp)
    return values


def _write_mujoco_relationship_check(
    *,
    scene_path: Path,
    diagnostics_root: Path,
    csd: ConcreteScenarioDefinition,
) -> tuple[str, tuple[CsdRealizationBlocker, ...]]:
    checks: list[dict[str, object]] = []
    blockers: list[CsdRealizationBlocker] = []
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for relationship in csd.relationships:
            if relationship.type == CsdRelationshipType.ON_TOP_OF:
                passed, details = _mujoco_on_top_of_check(
                    model,
                    data,
                    csd,
                    relationship,
                )
                checks.append(
                    _load_check(
                        f"on_top_of:{relationship.relation_id}",
                        passed=passed,
                        details=details,
                    )
                )
                if not passed:
                    blockers.append(
                        _csd_blocker(
                            csd.csd_id,
                            relationship.relation_id,
                            (
                                "MuJoCo initial state violates on_top_of relationship "
                                f"{relationship.relation_id}"
                            ),
                        )
                    )
                continue
            if relationship.type != CsdRelationshipType.AVOID_CONTACT:
                continue
            subject_body = _relationship_body_name(relationship.subject)
            object_body = _relationship_body_name(relationship.object)
            min_distance_m = float(relationship.parameters.get("min_distance_m", 0.0))
            subject_pos = _mujoco_body_world_position(model, data, subject_body)
            object_pos = _mujoco_body_world_position(model, data, object_body)
            distance_m = _distance(subject_pos, object_pos)
            passed = distance_m >= min_distance_m
            checks.append(
                _load_check(
                    f"avoid_contact:{relationship.relation_id}",
                    passed=passed,
                    details={
                        "subject": relationship.subject,
                        "object": relationship.object,
                        "distance_m": distance_m,
                        "min_distance_m": min_distance_m,
                    },
                )
            )
            if not passed:
                blockers.append(
                    _csd_blocker(
                        csd.csd_id,
                        relationship.relation_id,
                        (
                            "MuJoCo initial state violates avoid_contact relationship "
                            f"{relationship.relation_id}"
                        ),
                    )
                )
    except Exception as exc:
        checks.append(
            _load_check(
                "relationship_check",
                passed=False,
                details={"reason": str(exc)},
            )
        )
        blockers.append(
            CsdRealizationBlocker(
                blocker_id=f"{csd.csd_id}_{MUJOCO_BACKEND}_relationship_check_failed",
                csd_id=csd.csd_id,
                backend=MUJOCO_BACKEND,
                asset_id="relationships",
                scope="vsim_realization",
                reason="generated MuJoCo scene failed relationship diagnostics",
            )
        )

    payload = {
        "schema_version": "0.1",
        "backend": MUJOCO_BACKEND,
        "csd_id": csd.csd_id,
        "entry_file": scene_path.name,
        "status": "passed" if not blockers else "failed",
        "checks": checks,
    }
    path = diagnostics_root / "relationship_check.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "diagnostics/relationship_check.json", tuple(blockers)


def _mujoco_on_top_of_check(
    model: Any,
    data: Any,
    csd: ConcreteScenarioDefinition,
    relationship: CsdRelationship,
) -> tuple[bool, dict[str, object]]:
    subject_body = _relationship_body_name(relationship.subject)
    object_body = _relationship_body_name(relationship.object)
    subject_pos = _mujoco_body_world_position(model, data, subject_body)
    object_pos = _mujoco_body_world_position(model, data, object_body)
    tolerance_m = float(relationship.parameters.get("position_tolerance_m", 0.0))
    support_surface = _relationship_surface(csd, relationship.object)
    details: dict[str, object] = {
        "subject": relationship.subject,
        "object": relationship.object,
        "subject_position": subject_pos,
        "object_position": object_pos,
        "position_tolerance_m": _json_float(tolerance_m),
    }

    if support_surface is not None:
        dx = abs(subject_pos[0] - object_pos[0])
        dy = abs(subject_pos[1] - object_pos[1])
        max_dx = support_surface.size.x + tolerance_m
        max_dy = support_surface.size.y + tolerance_m
        min_subject_z = object_pos[2] + support_surface.size.z - tolerance_m
        passed = dx <= max_dx and dy <= max_dy and subject_pos[2] >= min_subject_z
        details.update(
            {
                "horizontal_delta_m": [_json_float(dx), _json_float(dy)],
                "horizontal_limit_m": [_json_float(max_dx), _json_float(max_dy)],
                "min_subject_z_m": _json_float(min_subject_z),
            }
        )
        return passed, details

    horizontal_distance_m = math.hypot(
        subject_pos[0] - object_pos[0],
        subject_pos[1] - object_pos[1],
    )
    support_radius_m = _mujoco_body_geom_radius(model, object_body)
    horizontal_limit_m = max(tolerance_m, support_radius_m)
    passed = (
        horizontal_distance_m <= horizontal_limit_m
        and subject_pos[2] >= object_pos[2] - tolerance_m
    )
    details.update(
        {
            "horizontal_distance_m": _json_float(horizontal_distance_m),
            "horizontal_limit_m": _json_float(horizontal_limit_m),
            "support_radius_m": _json_float(support_radius_m),
        }
    )
    return passed, details


def _relationship_body_name(entity_ref: str) -> str:
    entity_type, _, entity_id = entity_ref.partition(":")
    if entity_type in {"object", "surface"} and entity_id:
        return _mjcf_name(entity_id)
    raise ValueError(f"MuJoCo relationship diagnostics cannot resolve body for {entity_ref}")


def _relationship_surface(
    csd: ConcreteScenarioDefinition,
    entity_ref: str,
) -> CsdSurface | None:
    entity_type, _, entity_id = entity_ref.partition(":")
    if entity_type != "surface":
        return None
    return next(
        (
            surface
            for surface in csd.environment.surfaces
            if surface.surface_id == entity_id
        ),
        None,
    )


def _mujoco_body_world_position(model: Any, data: Any, body_name: str) -> list[float]:
    body_id = int(model.body(body_name).id)
    return _float_sequence_json(data.xpos[body_id])


def _mujoco_body_geom_radius(model: Any, body_name: str) -> float:
    body_id = int(model.body(body_name).id)
    radii = [
        float(model.geom_rbound[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
    ]
    return max(radii, default=0.0)


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _write_mujoco_physics_check(
    *,
    scene_path: Path,
    diagnostics_root: Path,
    csd: ConcreteScenarioDefinition,
    steps: int = 25,
) -> tuple[str, tuple[CsdRealizationBlocker, ...]]:
    checks: list[dict[str, object]] = []
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        checks.append(_load_check("mj_forward", passed=True))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        finite_state = _all_finite(data.qpos) and _all_finite(data.qvel)
        checks.append(
            _load_check(
                "finite_state_after_steps",
                passed=finite_state,
                details={
                    "steps": steps,
                    "nq": int(model.nq),
                    "nv": int(model.nv),
                },
            )
        )
    except Exception as exc:
        checks.append(
            _load_check(
                "mj_forward",
                passed=False,
                details={"reason": str(exc)},
            )
        )

    passed = all(check["status"] == "passed" for check in checks)
    payload = {
        "schema_version": "0.1",
        "backend": MUJOCO_BACKEND,
        "csd_id": csd.csd_id,
        "entry_file": scene_path.name,
        "status": "passed" if passed else "failed",
        "checks": checks,
    }
    path = diagnostics_root / "physics_check.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if passed:
        return "diagnostics/physics_check.json", ()
    return (
        "diagnostics/physics_check.json",
        (
            CsdRealizationBlocker(
                blocker_id=f"{csd.csd_id}_{MUJOCO_BACKEND}_physics_check_failed",
                csd_id=csd.csd_id,
                backend=MUJOCO_BACKEND,
                asset_id="scene",
                scope="vsim_realization",
                reason="generated MuJoCo scene failed short physics stability check",
            ),
        ),
    )


def _all_finite(values: Any) -> bool:
    return all(math.isfinite(float(value)) for value in values.flat)


def _write_mujoco_preview(
    *,
    scene_path: Path,
    diagnostics_root: Path,
    csd: ConcreteScenarioDefinition,
) -> tuple[str, tuple[CsdRealizationBlocker, ...]]:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        with mujoco.Renderer(
            model,
            height=MUJOCO_PREVIEW_SIZE_PX,
            width=MUJOCO_PREVIEW_SIZE_PX,
        ) as renderer:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=_mujoco_preview_camera(csd))
            pixels = renderer.render()
        if int(pixels.max()) <= int(pixels.min()):
            raise ValueError("rendered preview has no pixel variation")
        _write_ppm(diagnostics_root / "semantic_preview.ppm", pixels)
        return "diagnostics/semantic_preview.ppm", ()
    except Exception as exc:
        return (
            "diagnostics/semantic_preview.ppm",
            (
                CsdRealizationBlocker(
                    blocker_id=f"{csd.csd_id}_{MUJOCO_BACKEND}_preview_render_failed",
                    csd_id=csd.csd_id,
                    backend=MUJOCO_BACKEND,
                    asset_id="scene",
                    scope="vsim_realization",
                    reason=f"generated MuJoCo scene failed preview rendering: {exc}",
                ),
            ),
        )


def _mujoco_preview_camera(csd: ConcreteScenarioDefinition) -> str:
    if csd.environment.cameras:
        return _mjcf_name(csd.environment.cameras[0].camera_id)
    return "world_camera"


def _write_ppm(path: Path, pixels: Any) -> None:
    height, width = pixels.shape[:2]
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels.tobytes())


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
    inertial = obj.initial_state.inertial
    if inertial is not None:
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": _vector3_text(inertial.center_of_mass),
                "mass": _number_text(obj.initial_state.mass_kg),
                "diaginertia": _numbers_text(inertial.diagonal_inertia_kg_m2),
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
        visual_geom_attrs["density"] = "0"
        ET.SubElement(body, "geom", visual_geom_attrs)
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{_mjcf_name(obj.name)}_collision_geom",
                "type": "mesh",
                "mesh": f"{_mjcf_name(asset_id)}_collision",
                "friction": _numbers_text(obj.initial_state.friction),
                "rgba": "0 0 0 0",
                **_mujoco_geom_mass_attrs(obj),
                **_mujoco_contact_attrs(obj),
            },
        )
        return

    visual_geom_attrs["friction"] = _numbers_text(obj.initial_state.friction)
    visual_geom_attrs.update(_mujoco_geom_mass_attrs(obj))
    visual_geom_attrs.update(_mujoco_contact_attrs(obj))
    ET.SubElement(body, "geom", visual_geom_attrs)


def _mujoco_object_physical_blockers(
    csd_id: str,
    obj: CsdObject,
) -> tuple[CsdRealizationBlocker, ...]:
    blockers: list[CsdRealizationBlocker] = []
    if obj.initial_state.mass_kg <= 0.0:
        blockers.append(
            _object_csd_blocker(
                csd_id,
                obj.name,
                "mass_kg",
                f"object {obj.name} mass_kg must be positive",
            )
        )
    if any(value < 0.0 for value in obj.initial_state.friction):
        blockers.append(
            _object_csd_blocker(
                csd_id,
                obj.name,
                "friction",
                f"object {obj.name} friction values must be non-negative",
            )
        )
    inertial = obj.initial_state.inertial
    if inertial is not None:
        if not _finite_vector(inertial.center_of_mass):
            blockers.append(
                _object_csd_blocker(
                    csd_id,
                    obj.name,
                    "inertial",
                    f"object {obj.name} inertial center_of_mass values must be finite",
                )
            )
        if any(
            not math.isfinite(float(value)) or value <= 0.0
            for value in inertial.diagonal_inertia_kg_m2
        ):
            blockers.append(
                _object_csd_blocker(
                    csd_id,
                    obj.name,
                    "inertial",
                    (
                        f"object {obj.name} diagonal_inertia_kg_m2 values "
                        "must be positive and finite"
                    ),
                )
            )
    contact = obj.initial_state.contact
    if contact is None:
        return tuple(blockers)
    if contact.margin_m is not None and contact.margin_m < 0.0:
        blockers.append(
            _object_csd_blocker(
                csd_id,
                obj.name,
                "contact",
                f"object {obj.name} contact margin_m must be non-negative",
            )
        )
    if contact.gap_m is not None and contact.gap_m < 0.0:
        blockers.append(
            _object_csd_blocker(
                csd_id,
                obj.name,
                "contact",
                f"object {obj.name} contact gap_m must be non-negative",
            )
        )
    if (
        contact.margin_m is not None
        and contact.gap_m is not None
        and contact.gap_m > contact.margin_m
    ):
        blockers.append(
            _object_csd_blocker(
                csd_id,
                obj.name,
                "contact",
                f"object {obj.name} contact gap_m must be less than or equal to margin_m",
            )
        )
    return tuple(blockers)


def _mujoco_geom_mass_attrs(obj: CsdObject) -> dict[str, str]:
    if obj.initial_state.inertial is not None:
        return {"density": "0"}
    return {"mass": _number_text(obj.initial_state.mass_kg)}


def _mujoco_contact_attrs(obj: CsdObject) -> dict[str, str]:
    contact = obj.initial_state.contact
    if contact is None:
        return {}
    attrs: dict[str, str] = {}
    if contact.margin_m is not None:
        attrs["margin"] = _number_text(contact.margin_m)
    if contact.gap_m is not None:
        attrs["gap"] = _number_text(contact.gap_m)
    if contact.solref is not None:
        attrs["solref"] = _numbers_text(contact.solref)
    if contact.solimp is not None:
        attrs["solimp"] = _numbers_text(contact.solimp)
    return attrs


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
        if not _safe_relative_resource_path(relative_path):
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"backend resource path must stay inside asset root: {relative_path}",
                )
            )
            continue
        if backend == MUJOCO_BACKEND and not _is_supported_mujoco_mesh_path(relative_path):
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"MuJoCo mesh resource format is unsupported: {relative_path}",
                )
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
            elif not _safe_relative_resource_path(collision_mesh_path):
                blockers.append(
                    _asset_blocker(
                        csd_id,
                        asset_id,
                        backend,
                        (
                            "asset collision mesh path must stay inside asset root: "
                            f"{collision_mesh_path}"
                        ),
                    )
                )
            elif backend == MUJOCO_BACKEND and not _is_supported_mujoco_mesh_path(
                collision_mesh_path
            ):
                blockers.append(
                    _asset_blocker(
                        csd_id,
                        asset_id,
                        backend,
                        (
                            "MuJoCo collision mesh resource format is unsupported: "
                            f"{collision_mesh_path}"
                        ),
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
        if not _safe_relative_resource_path(texture_path):
            blockers.append(
                _asset_blocker(
                    csd_id,
                    asset_id,
                    backend,
                    f"asset material texture path must stay inside asset root: {texture_path}",
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


def _is_supported_mujoco_mesh_path(path: str) -> bool:
    return Path(path).suffix.lower() in MUJOCO_MESH_EXTENSIONS


def _safe_relative_resource_path(path: str) -> bool:
    resource_path = Path(path)
    return bool(path) and not resource_path.is_absolute() and ".." not in resource_path.parts


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


def _object_csd_blocker(
    csd_id: str,
    object_name: str,
    field_name: str,
    reason: str,
) -> CsdRealizationBlocker:
    return CsdRealizationBlocker(
        blocker_id=(
            f"{csd_id}_{MUJOCO_BACKEND}_{_mjcf_name(object_name)}_"
            f"{_mjcf_name(field_name)}_compile_blocked"
        ),
        csd_id=csd_id,
        backend=MUJOCO_BACKEND,
        asset_id=object_name,
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


def _vector3_json(vector: Any) -> list[float]:
    return [_json_float(vector.x), _json_float(vector.y), _json_float(vector.z)]


def _quaternion_json(quaternion: Any) -> list[float]:
    values = (float(quaternion.w), float(quaternion.x), float(quaternion.y), float(quaternion.z))
    norm = _quaternion_norm(quaternion)
    if norm == 0.0:
        return [_orientation_float(value) for value in values]
    return [_orientation_float(value / norm) for value in values]


def _quaternion_norm(quaternion: Any) -> float:
    return math.sqrt(
        float(quaternion.w) * float(quaternion.w)
        + float(quaternion.x) * float(quaternion.x)
        + float(quaternion.y) * float(quaternion.y)
        + float(quaternion.z) * float(quaternion.z)
    )


def _quaternion_sequence_json(values: Any) -> list[float]:
    return [_orientation_float(value) for value in values]


def _rgba_sequence_json(values: Any) -> list[float]:
    return [_orientation_float(value) for value in values]


def _camera_xyaxes_quaternion_json(
    xyaxes: tuple[float, float, float, float, float, float],
) -> list[float]:
    x_axis = _unit_vector((xyaxes[0], xyaxes[1], xyaxes[2]))
    y_axis = _unit_vector((xyaxes[3], xyaxes[4], xyaxes[5]))
    z_axis = _unit_vector(_cross(x_axis, y_axis))
    matrix = (
        x_axis[0],
        y_axis[0],
        z_axis[0],
        x_axis[1],
        y_axis[1],
        z_axis[1],
        x_axis[2],
        y_axis[2],
        z_axis[2],
    )
    return [_orientation_float(value) for value in _matrix_to_quaternion(matrix)]


def _valid_camera_xyaxes(
    xyaxes: tuple[float, float, float, float, float, float],
) -> bool:
    x_axis = (xyaxes[0], xyaxes[1], xyaxes[2])
    y_axis = (xyaxes[3], xyaxes[4], xyaxes[5])
    if _vector_norm(x_axis) == 0.0 or _vector_norm(y_axis) == 0.0:
        return False
    return _vector_norm(_cross(x_axis, y_axis)) > 0.0


def _unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _vector_norm(vector)
    if norm == 0.0:
        raise ValueError("camera xyaxes must not contain a zero axis")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _positive_finite_vector(vector: Any) -> bool:
    return all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (vector.x, vector.y, vector.z)
    )


def _finite_vector(vector: Any) -> bool:
    return all(math.isfinite(float(value)) for value in (vector.x, vector.y, vector.z))


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _matrix_to_quaternion(
    matrix: tuple[float, float, float, float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = matrix
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (r21 - r12) / scale,
            (r02 - r20) / scale,
            (r10 - r01) / scale,
        )
    if r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return (
            (r21 - r12) / scale,
            0.25 * scale,
            (r01 + r10) / scale,
            (r02 + r20) / scale,
        )
    if r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return (
            (r02 - r20) / scale,
            (r01 + r10) / scale,
            0.25 * scale,
            (r12 + r21) / scale,
        )
    scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return (
        (r10 - r01) / scale,
        (r02 + r20) / scale,
        (r12 + r21) / scale,
        0.25 * scale,
    )


def _float_sequence_json(values: Any) -> list[float]:
    return [_json_float(value) for value in values]


def _json_float(value: Any) -> float:
    return round(float(value), 12)


def _orientation_float(value: Any) -> float:
    return round(float(value), 6)


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
