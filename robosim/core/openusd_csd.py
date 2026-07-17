"""OpenUSD CSD schema registration and stage helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pxr import Plug, Sdf, Usd, UsdGeom, UsdUtils

_PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "usd_plugins" / "robosimCsd"
_BACKENDS = frozenset({"mujoco", "pybullet", "gazebo"})
_RELATIONSHIP_TYPES = frozenset(
    {
        "on_top_of",
        "inside",
        "near",
        "avoid_contact",
        "aligned_with",
        "attached_to",
    }
)
_SENSOR_TYPES = frozenset(
    {"rgb", "depth", "joint_state", "imu", "lidar", "odometry", "force_torque"}
)
_RANDOMIZATION_PREFIX = "robosim:randomization:value:"


@dataclass(frozen=True, slots=True)
class CsdStageValidationIssue:
    """One deterministic semantic validation failure in a CSD stage."""

    code: str
    prim_path: Sdf.Path
    message: str


@dataclass(frozen=True, slots=True)
class OpenUsdEntity:
    """Stable CSD entity identity read from an OpenUSD prim."""

    prim_path: Sdf.Path
    entity_id: str
    asset_id: str
    role: str
    static: bool
    world_transform: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OpenUsdRelationship:
    """Typed relationship read from a CSD stage."""

    prim_path: Sdf.Path
    relationship_type: str
    subject: Sdf.Path
    object: Sdf.Path
    min_distance_m: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class OpenUsdSensorRequirement:
    """Typed runtime sensor requirement read from a CSD stage."""

    prim_path: Sdf.Path
    sensor_type: str
    requirement: str
    min_resolution: tuple[int, int]
    source: Sdf.Path


@dataclass(frozen=True, slots=True)
class OpenUsdEvaluatorRef:
    """Resolved evaluator artifact reference read from a CSD stage."""

    prim_path: Sdf.Path
    artifact_id: str
    kind: str
    path: Path


@dataclass(frozen=True, slots=True)
class OpenUsdCsd:
    """Validated composed CSD and its typed project handoff semantics."""

    root_path: Path
    stage: Usd.Stage
    backend: str
    digest: str
    csd_id: str
    schema_version: str
    task_instance_id: str
    world_template_id: str
    task_objective: str
    randomization_seed: int
    randomization_values: dict[str, str | int | float | bool]
    entities: tuple[OpenUsdEntity, ...]
    relationships: tuple[OpenUsdRelationship, ...]
    sensors: tuple[OpenUsdSensorRequirement, ...]
    evaluators: tuple[OpenUsdEvaluatorRef, ...]


def csd_plugin_root() -> Path:
    """Return the packaged codeless CSD schema resource directory."""
    return _PLUGIN_ROOT


def register_csd_plugins() -> None:
    """Register the packaged codeless CSD schemas with OpenUSD."""
    Plug.Registry().RegisterPlugins(str(_PLUGIN_ROOT / "plugInfo.json"))


def compute_csd_digest(csd_path: Path) -> str:
    """Hash a CSD's composed authored layers and resolved non-USD dependencies."""
    root_path = Path(csd_path).resolve()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root_path))
    if unresolved:
        missing = ", ".join(sorted(str(path) for path in unresolved))
        raise ValueError(f"unresolved CSD dependencies: {missing}")

    entries: list[tuple[str, bytes]] = []
    for layer in layers:
        layer_path = Path(layer.realPath or layer.identifier).resolve()
        entries.append((_dependency_label(layer_path, root_path.parent), layer_path.read_bytes()))
    for asset in assets:
        asset_path = Path(str(asset)).resolve()
        entries.append((_dependency_label(asset_path, root_path.parent), asset_path.read_bytes()))

    digest = hashlib.sha256()
    for label, content in sorted(entries):
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def read_openusd_csd(csd_path: Path, *, backend: str) -> OpenUsdCsd:
    """Open, select, validate, and read one canonical composed CSD stage."""
    backend_key = backend.strip().lower()
    if backend_key not in _BACKENDS:
        raise ValueError(f"unsupported CSD backend: {backend}")

    register_csd_plugins()
    root_path = Path(csd_path).resolve()
    stage = Usd.Stage.Open(str(root_path))
    if stage is None:
        raise ValueError(f"failed to open CSD stage: {root_path}")
    stage.SetEditTarget(stage.GetSessionLayer())
    variants = stage.GetDefaultPrim().GetVariantSet("physicsBackend")
    if not variants.SetVariantSelection(backend_key):
        raise ValueError(f"CSD has no physicsBackend variant: {backend_key}")

    issues = validate_csd_stage(stage)
    if issues:
        details = "; ".join(f"{issue.code} at {issue.prim_path}" for issue in issues)
        raise ValueError(f"invalid CSD stage: {details}")

    world = stage.GetDefaultPrim()
    task = stage.GetPrimAtPath("/World/Task")
    randomization_values = {
        attr.GetName().removeprefix(_RANDOMIZATION_PREFIX): _scalar_value(attr.Get())
        for attr in world.GetAttributes()
        if attr.GetName().startswith(_RANDOMIZATION_PREFIX)
    }
    xform_cache = UsdGeom.XformCache()
    entities = tuple(
        OpenUsdEntity(
            prim_path=prim.GetPath(),
            entity_id=str(prim.GetAttribute("robosim:entity:id").Get()),
            asset_id=str(prim.GetAttribute("robosim:entity:assetId").Get()),
            role=str(prim.GetAttribute("robosim:entity:role").Get()),
            static=bool(prim.GetAttribute("robosim:entity:static").Get()),
            world_transform=_flatten_matrix(xform_cache.GetLocalToWorldTransform(prim)),
        )
        for prim in stage.Traverse()
        if _has_applied_schema(prim, "RobosimEntityAPI")
    )
    relationships = tuple(
        _read_relationship(prim) for prim in _prims_of_type(stage, "RobosimRelationship")
    )
    sensors = tuple(
        _read_sensor(prim) for prim in _prims_of_type(stage, "RobosimSensorRequirement")
    )
    evaluators = tuple(
        _read_evaluator(prim) for prim in _prims_of_type(stage, "RobosimEvaluatorRef")
    )
    return OpenUsdCsd(
        root_path=root_path,
        stage=stage,
        backend=backend_key,
        digest=compute_csd_digest(root_path),
        csd_id=str(world.GetAttribute("robosim:csd:id").Get()),
        schema_version=str(world.GetAttribute("robosim:csd:schemaVersion").Get()),
        task_instance_id=str(world.GetAttribute("robosim:csd:taskInstanceId").Get()),
        world_template_id=str(world.GetAttribute("robosim:csd:worldTemplateId").Get()),
        task_objective=str(task.GetAttribute("robosim:task:objective").Get()),
        randomization_seed=int(world.GetAttribute("robosim:randomization:seed").Get()),
        randomization_values=randomization_values,
        entities=entities,
        relationships=relationships,
        sensors=sensors,
        evaluators=evaluators,
    )


def validate_csd_stage(stage: Usd.Stage) -> tuple[CsdStageValidationIssue, ...]:
    """Validate project semantics that `usdchecker` cannot infer."""
    register_csd_plugins()
    issues: list[CsdStageValidationIssue] = []
    world = stage.GetDefaultPrim()
    if not world or world.GetPath() != Sdf.Path("/World"):
        return (
            _issue(
                "invalid_default_prim", Sdf.Path.absoluteRootPath, "default prim must be /World"
            ),
        )
    if UsdGeom.GetStageMetersPerUnit(stage) <= 0:
        issues.append(
            _issue("invalid_stage_units", world.GetPath(), "metersPerUnit must be positive")
        )
    if UsdGeom.GetStageUpAxis(stage) not in {UsdGeom.Tokens.y, UsdGeom.Tokens.z}:
        issues.append(_issue("invalid_up_axis", world.GetPath(), "upAxis must be Y or Z"))
    if not _has_applied_schema(world, "RobosimCsdRootAPI"):
        issues.append(
            _issue("missing_root_schema", world.GetPath(), "RobosimCsdRootAPI is required")
        )
    for name in (
        "robosim:csd:id",
        "robosim:csd:schemaVersion",
        "robosim:csd:taskInstanceId",
        "robosim:csd:worldTemplateId",
    ):
        if not str(world.GetAttribute(name).Get() or ""):
            issues.append(_issue("missing_root_value", world.GetPath(), f"{name} is required"))

    variants = world.GetVariantSet("physicsBackend")
    if set(variants.GetVariantNames()) != _BACKENDS:
        issues.append(
            _issue(
                "invalid_backend_variants",
                world.GetPath(),
                "physicsBackend must define mujoco, pybullet, and gazebo",
            )
        )
    for path in ("/World/Task", "/World/PhysicsScene", "/World/Objects"):
        if not stage.GetPrimAtPath(path):
            issues.append(
                _issue("missing_required_prim", Sdf.Path(path), f"required prim is missing: {path}")
            )

    entity_paths: dict[str, Sdf.Path] = {}
    for prim in stage.Traverse():
        if not _has_applied_schema(prim, "RobosimEntityAPI"):
            continue
        entity_id = str(prim.GetAttribute("robosim:entity:id").Get() or "")
        asset_id = str(prim.GetAttribute("robosim:entity:assetId").Get() or "")
        if not entity_id or not asset_id:
            issues.append(
                _issue("invalid_entity", prim.GetPath(), "entity ID and asset ID are required")
            )
        elif entity_id in entity_paths:
            issues.append(
                _issue(
                    "duplicate_entity_id",
                    prim.GetPath(),
                    f"entity ID {entity_id!r} is already used at {entity_paths[entity_id]}",
                )
            )
        else:
            entity_paths[entity_id] = prim.GetPath()

    for attr in world.GetAttributes():
        if not attr.GetName().startswith(_RANDOMIZATION_PREFIX):
            continue
        if attr.ValueMightBeTimeVarying() or attr.GetTypeName().isArray:
            issues.append(
                _issue(
                    "nonconcrete_randomization",
                    world.GetPath(),
                    f"randomization value must be one fixed scalar: {attr.GetName()}",
                )
            )

    for prim in _prims_of_type(stage, "RobosimRelationship"):
        relationship_type = str(prim.GetAttribute("robosim:relationship:type").Get() or "")
        if relationship_type not in _RELATIONSHIP_TYPES:
            issues.append(_issue("invalid_relationship_type", prim.GetPath(), relationship_type))
        for name in ("minDistanceM", "tolerance"):
            value = _float_attribute(prim, f"robosim:relationship:{name}")
            if not math.isfinite(value) or value < 0:
                issues.append(
                    _issue(
                        "invalid_relationship_parameter",
                        prim.GetPath(),
                        f"{name} must be finite and nonnegative",
                    )
                )
        for name in ("subject", "object"):
            targets = prim.GetRelationship(f"robosim:relationship:{name}").GetTargets()
            if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
                issues.append(
                    _issue(
                        "unresolved_relationship_target",
                        prim.GetPath(),
                        f"relationship {name} must resolve to one prim",
                    )
                )

    for prim in _prims_of_type(stage, "RobosimSensorRequirement"):
        sensor_type = str(prim.GetAttribute("robosim:sensor:type").Get() or "")
        if sensor_type not in _SENSOR_TYPES:
            issues.append(_issue("invalid_sensor_type", prim.GetPath(), sensor_type))
        requirement = str(prim.GetAttribute("robosim:sensor:requirement").Get() or "")
        if requirement not in {"required", "optional"}:
            issues.append(_issue("invalid_sensor_requirement", prim.GetPath(), requirement))
        resolution = prim.GetAttribute("robosim:sensor:minResolution").Get()
        if sensor_type in {"rgb", "depth"} and (resolution[0] <= 0 or resolution[1] <= 0):
            issues.append(
                _issue("invalid_sensor_resolution", prim.GetPath(), "image resolution is required")
            )
        targets = prim.GetRelationship("robosim:sensor:source").GetTargets()
        if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
            issues.append(
                _issue("unresolved_sensor_source", prim.GetPath(), "sensor source must resolve")
            )

    for prim in _prims_of_type(stage, "RobosimEvaluatorRef"):
        if not str(prim.GetAttribute("robosim:evaluator:artifactId").Get() or ""):
            issues.append(_issue("invalid_evaluator", prim.GetPath(), "artifact ID is required"))
        if not str(prim.GetAttribute("robosim:evaluator:kind").Get() or ""):
            issues.append(_issue("invalid_evaluator", prim.GetPath(), "kind is required"))
        evaluator_path = prim.GetAttribute("robosim:evaluator:path").Get()
        if not isinstance(evaluator_path, Sdf.AssetPath) or not evaluator_path.resolvedPath:
            issues.append(
                _issue("unresolved_evaluator", prim.GetPath(), "evaluator path must resolve")
            )
    return tuple(issues)


def _dependency_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _flatten_matrix(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _has_applied_schema(prim: Usd.Prim, schema_name: str) -> bool:
    schemas = prim.GetMetadata("apiSchemas")
    return bool(schemas and schema_name in schemas.GetAppliedItems())


def _float_attribute(prim: Usd.Prim, name: str, default: float = 0.0) -> float:
    value = prim.GetAttribute(name).Get()
    return default if value is None else float(value)


def _prims_of_type(stage: Usd.Stage, type_name: str) -> tuple[Usd.Prim, ...]:
    return tuple(prim for prim in stage.Traverse() if prim.GetTypeName() == type_name)


def _single_target(prim: Usd.Prim, name: str) -> Sdf.Path:
    return prim.GetRelationship(name).GetTargets()[0]


def _read_relationship(prim: Usd.Prim) -> OpenUsdRelationship:
    return OpenUsdRelationship(
        prim_path=prim.GetPath(),
        relationship_type=str(prim.GetAttribute("robosim:relationship:type").Get()),
        subject=_single_target(prim, "robosim:relationship:subject"),
        object=_single_target(prim, "robosim:relationship:object"),
        min_distance_m=_float_attribute(prim, "robosim:relationship:minDistanceM"),
        tolerance=_float_attribute(prim, "robosim:relationship:tolerance"),
    )


def _read_sensor(prim: Usd.Prim) -> OpenUsdSensorRequirement:
    resolution = prim.GetAttribute("robosim:sensor:minResolution").Get()
    return OpenUsdSensorRequirement(
        prim_path=prim.GetPath(),
        sensor_type=str(prim.GetAttribute("robosim:sensor:type").Get()),
        requirement=str(prim.GetAttribute("robosim:sensor:requirement").Get()),
        min_resolution=(int(resolution[0]), int(resolution[1])),
        source=_single_target(prim, "robosim:sensor:source"),
    )


def _read_evaluator(prim: Usd.Prim) -> OpenUsdEvaluatorRef:
    asset_path = prim.GetAttribute("robosim:evaluator:path").Get()
    return OpenUsdEvaluatorRef(
        prim_path=prim.GetPath(),
        artifact_id=str(prim.GetAttribute("robosim:evaluator:artifactId").Get()),
        kind=str(prim.GetAttribute("robosim:evaluator:kind").Get()),
        path=Path(asset_path.resolvedPath),
    )


def _scalar_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"CSD randomization value is not a scalar: {value!r}")


def _issue(code: str, prim_path: Sdf.Path, message: str) -> CsdStageValidationIssue:
    return CsdStageValidationIssue(code=code, prim_path=prim_path, message=message)
