"""OpenUSD CSD schema registration and stage helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdUtils

from robosim.core.csd import (
    DEFAULT_MUJOCO_OBJECT_FRICTION,
    ConcreteScenarioDefinition,
    CsdCamera,
    CsdEnvironment,
    CsdLight,
    CsdObject,
    CsdObjectContact,
    CsdObjectInertial,
    CsdObjectInitialState,
    CsdPose,
    CsdQuaternion,
    CsdRelationship,
    CsdRelationshipType,
    CsdRobot,
    CsdSurface,
    CsdVector3,
)

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


def compiler_csd_from_openusd(csd: OpenUsdCsd) -> ConcreteScenarioDefinition:
    """Build the backend compiler's typed view directly from a composed stage."""
    stage = csd.stage
    xforms = UsdGeom.XformCache()
    entities = {entity.prim_path: entity for entity in csd.entities}
    environment_entity = next(
        (entity for entity in csd.entities if entity.role == "environment"),
        None,
    )
    robot_entity = next(
        (entity for entity in csd.entities if entity.role == "robot"),
        None,
    )
    objects = tuple(
        _compiler_object(stage.GetPrimAtPath(entity.prim_path), entity, xforms)
        for entity in csd.entities
        if entity.prim_path.HasPrefix(Sdf.Path("/World/Objects"))
    )
    surfaces = tuple(
        surface
        for entity in csd.entities
        if entity.role == "support_surface"
        for surface in (_compiler_surface(stage.GetPrimAtPath(entity.prim_path), entity, xforms),)
        if surface is not None
    )
    relationships = tuple(
        _compiler_relationship(relationship, entities) for relationship in csd.relationships
    )
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    return ConcreteScenarioDefinition(
        csd_id=csd.csd_id,
        schema_version=csd.schema_version,
        frame="world",
        units="m" if math.isclose(meters_per_unit, 1.0) else f"{meters_per_unit:g}m",
        environment=CsdEnvironment(
            environment_id=(
                environment_entity.asset_id
                if environment_entity is not None
                else csd.world_template_id
            ),
            environment_type="openusd",
            gravity=_compiler_gravity(stage),
            surfaces=surfaces,
            cameras=_compiler_cameras(stage, xforms),
            lighting=_compiler_lights(stage, xforms),
        ),
        robot=(
            CsdRobot(
                asset_id=robot_entity.asset_id,
                pose=_compiler_pose(stage.GetPrimAtPath(robot_entity.prim_path), xforms),
            )
            if robot_entity is not None
            else None
        ),
        objects=objects,
        relationships=relationships,
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

    for prim in stage.Traverse():
        if not UsdGeom.Xformable(prim):
            continue
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeOrient:
                continue
            quaternion = op.Get()
            if quaternion is None:
                continue
            values = (quaternion.GetReal(), *quaternion.GetImaginary())
            if not all(math.isfinite(float(value)) for value in values) or math.isclose(
                sum(float(value) ** 2 for value in values), 0.0
            ):
                issues.append(
                    _issue(
                        "invalid_xform_orientation",
                        prim.GetPath(),
                        "orientation quaternion must be finite and nonzero",
                    )
                )

    for prim in stage.Traverse():
        diagonal = prim.GetAttribute("physics:diagonalInertia")
        principal_axes = prim.GetAttribute("physics:principalAxes")
        has_diagonal = bool(diagonal) and diagonal.HasAuthoredValueOpinion()
        has_principal_axes = bool(principal_axes) and principal_axes.HasAuthoredValueOpinion()
        if has_diagonal != has_principal_axes:
            issues.append(
                _issue(
                    "invalid_inertia_pair",
                    prim.GetPath(),
                    "physics:diagonalInertia and physics:principalAxes must be authored together",
                )
            )
            continue
        if not has_principal_axes:
            continue
        quaternion = principal_axes.Get()
        values = (quaternion.GetReal(), *quaternion.GetImaginary())
        norm = math.sqrt(sum(float(value) ** 2 for value in values))
        if (
            not all(math.isfinite(float(value)) for value in values)
            or math.isclose(norm, 0.0)
            or not all(math.isclose(float(value), 0.0, abs_tol=1e-6) for value in values[1:])
        ):
            issues.append(
                _issue(
                    "unsupported_principal_axes",
                    prim.GetPath(),
                    "backend compilers currently require identity physics:principalAxes",
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


def _compiler_pose(prim: Usd.Prim, xforms: UsdGeom.XformCache) -> CsdPose:
    transform = Gf.Transform(xforms.GetLocalToWorldTransform(prim))
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return CsdPose(
        position=CsdVector3(*map(float, translation)),
        orientation=CsdQuaternion(
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _compiler_object(
    prim: Usd.Prim,
    entity: OpenUsdEntity,
    xforms: UsdGeom.XformCache,
) -> CsdObject:
    mass = _attribute_float(prim, "physics:mass", 0.1)
    diagonal = prim.GetAttribute("physics:diagonalInertia").Get()
    center = prim.GetAttribute("physics:centerOfMass").Get()
    inertial = None
    if diagonal is not None and any(float(value) != 0.0 for value in diagonal):
        center = center or Gf.Vec3f(0.0)
        inertial = CsdObjectInertial(
            center_of_mass=CsdVector3(*(_clean_float(value) for value in center)),
            diagonal_inertia_kg_m2=(
                _clean_float(diagonal[0]),
                _clean_float(diagonal[1]),
                _clean_float(diagonal[2]),
            ),
        )
    return CsdObject(
        name=entity.entity_id,
        asset_id=entity.asset_id,
        role=entity.role,
        pose=_compiler_pose(prim, xforms),
        static=entity.static,
        initial_state=CsdObjectInitialState(
            mass_kg=mass,
            friction=_compiler_friction(prim, DEFAULT_MUJOCO_OBJECT_FRICTION),
            contact=_compiler_contact(prim),
            inertial=inertial,
        ),
        rgba=_compiler_object_rgba(prim),
    )


def _compiler_surface(
    prim: Usd.Prim,
    entity: OpenUsdEntity,
    xforms: UsdGeom.XformCache,
) -> CsdSurface | None:
    if prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        authored_size = float(cube.GetSizeAttr().Get() or 1.0)
        scale = Gf.Transform(xforms.GetLocalToWorldTransform(prim)).GetScale()
        half_extents = tuple(_clean_float(float(value) * authored_size / 2.0) for value in scale)
        surface_type = "box"
    elif prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        radius = _clean_float(cylinder.GetRadiusAttr().Get() or 1.0)
        half_height = _clean_float(float(cylinder.GetHeightAttr().Get() or 2.0) / 2.0)
        half_extents = (radius, radius, half_height)
        surface_type = "cylinder"
    else:
        return None
    colors = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get() or ()
    color = colors[0] if colors else Gf.Vec3f(0.42, 0.36, 0.28)
    opacities = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get() or ()
    opacity = _clean_float(opacities[0]) if opacities else 1.0
    rgba = (
        _clean_float(color[0]),
        _clean_float(color[1]),
        _clean_float(color[2]),
        opacity,
    )
    return CsdSurface(
        surface_id=entity.entity_id,
        surface_type=surface_type,
        pose=_compiler_pose(prim, xforms),
        size=CsdVector3(*half_extents),
        rgba=rgba,
        friction=_compiler_friction(prim, (1.2, 0.2, 0.2)),
    )


def _compiler_gravity(stage: Usd.Stage) -> CsdVector3:
    scene = stage.GetPrimAtPath("/World/PhysicsScene")
    direction = scene.GetAttribute("physics:gravityDirection").Get() or Gf.Vec3f(0.0, 0.0, -1.0)
    magnitude = _attribute_float(scene, "physics:gravityMagnitude", 9.81)
    return CsdVector3(*(_clean_float(float(value) * magnitude) for value in direction))


def _compiler_cameras(
    stage: Usd.Stage,
    xforms: UsdGeom.XformCache,
) -> tuple[CsdCamera, ...]:
    cameras: list[CsdCamera] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        transform = Gf.Transform(xforms.GetLocalToWorldTransform(prim))
        translation = transform.GetTranslation()
        rotation = Gf.Matrix3d(transform.GetRotation())
        cameras.append(
            CsdCamera(
                camera_id=prim.GetName(),
                position=CsdVector3(*map(float, translation)),
                xyaxes=(
                    float(rotation[0][0]),
                    float(rotation[0][1]),
                    float(rotation[0][2]),
                    float(rotation[1][0]),
                    float(rotation[1][1]),
                    float(rotation[1][2]),
                ),
                mode="fixed",
            )
        )
    return tuple(cameras)


def _compiler_lights(
    stage: Usd.Stage,
    xforms: UsdGeom.XformCache,
) -> tuple[CsdLight, ...]:
    lights: list[CsdLight] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.BoundableLightBase) and not prim.IsA(UsdLux.NonboundableLightBase):
            continue
        transform = Gf.Transform(xforms.GetLocalToWorldTransform(prim))
        translation = transform.GetTranslation()
        direction = transform.GetRotation().TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        lights.append(
            CsdLight(
                light_id=prim.GetName(),
                position=CsdVector3(*map(float, translation)),
                direction=CsdVector3(*map(float, direction)),
            )
        )
    return tuple(lights)


def _compiler_relationship(
    relationship: OpenUsdRelationship,
    entities: dict[Sdf.Path, OpenUsdEntity],
) -> CsdRelationship:
    return CsdRelationship(
        relation_id=relationship.prim_path.name,
        type=CsdRelationshipType(relationship.relationship_type),
        subject=_compiler_entity_ref(relationship.subject, entities),
        object=_compiler_entity_ref(relationship.object, entities),
        parameters={
            "min_distance_m": relationship.min_distance_m,
            "position_tolerance_m": relationship.tolerance,
        },
    )


def _compiler_entity_ref(
    path: Sdf.Path,
    entities: dict[Sdf.Path, OpenUsdEntity],
) -> str:
    entity = entities[path]
    if entity.role == "support_surface":
        return f"surface:{entity.entity_id}"
    if entity.role == "robot":
        return f"robot:{entity.asset_id}"
    return f"object:{entity.entity_id}"


def _compiler_friction(
    prim: Usd.Prim,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    mujoco_friction = prim.GetAttribute("robosim:mujoco:friction").Get()
    if mujoco_friction is not None:
        if len(mujoco_friction) != 3:
            raise ValueError("robosim:mujoco:friction must contain three values")
        return (
            _clean_float(mujoco_friction[0]),
            _clean_float(mujoco_friction[1]),
            _clean_float(mujoco_friction[2]),
        )
    for candidate in Usd.PrimRange(prim):
        material, _ = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()
        if not material:
            continue
        material_prim = material.GetPrim()
        dynamic = material_prim.GetAttribute("physics:dynamicFriction").Get()
        static = material_prim.GetAttribute("physics:staticFriction").Get()
        sliding = dynamic if dynamic is not None else static
        if sliding is not None:
            return (_clean_float(sliding), default[1], default[2])
    return default


def _compiler_object_rgba(prim: Usd.Prim) -> tuple[float, float, float, float] | None:
    for candidate in Usd.PrimRange(prim):
        material, _ = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()
        if material:
            shader = material.ComputeSurfaceSource()[0]
            if shader:
                diffuse = shader.GetInput("diffuseColor").Get()
                if diffuse is not None:
                    opacity = shader.GetInput("opacity").Get()
                    return (
                        _clean_float(diffuse[0]),
                        _clean_float(diffuse[1]),
                        _clean_float(diffuse[2]),
                        _clean_float(opacity) if opacity is not None else 1.0,
                    )
        if candidate.IsA(UsdGeom.Gprim):
            colors = UsdGeom.Gprim(candidate).GetDisplayColorAttr().Get() or ()
            if colors:
                opacities = UsdGeom.Gprim(candidate).GetDisplayOpacityAttr().Get() or ()
                return (
                    _clean_float(colors[0][0]),
                    _clean_float(colors[0][1]),
                    _clean_float(colors[0][2]),
                    _clean_float(opacities[0]) if opacities else 1.0,
                )
    return None


def _compiler_contact(prim: Usd.Prim) -> CsdObjectContact | None:
    candidate = next(
        (
            item
            for item in Usd.PrimRange(prim)
            if any(
                item.HasAttribute(name)
                for name in ("mjc:margin", "mjc:gap", "mjc:solref", "mjc:solimp")
            )
        ),
        None,
    )
    if candidate is None:
        return None
    solref = candidate.GetAttribute("mjc:solref").Get()
    solimp = candidate.GetAttribute("mjc:solimp").Get()
    if solref is not None and len(solref) != 2:
        raise ValueError("mjc:solref must contain two values")
    if solimp is not None and len(solimp) != 5:
        raise ValueError("mjc:solimp must contain five values")
    return CsdObjectContact(
        margin_m=_optional_attribute_float(candidate, "mjc:margin"),
        gap_m=_optional_attribute_float(candidate, "mjc:gap"),
        solref=(_clean_float(solref[0]), _clean_float(solref[1])) if solref is not None else None,
        solimp=(
            (
                _clean_float(solimp[0]),
                _clean_float(solimp[1]),
                _clean_float(solimp[2]),
                _clean_float(solimp[3]),
                _clean_float(solimp[4]),
            )
            if solimp is not None
            else None
        ),
    )


def _attribute_float(prim: Usd.Prim, name: str, default: float) -> float:
    value = prim.GetAttribute(name).Get()
    return default if value is None else _clean_float(value)


def _optional_attribute_float(prim: Usd.Prim, name: str) -> float | None:
    value = prim.GetAttribute(name).Get()
    return None if value is None else _clean_float(value)


def _clean_float(value: Any) -> float:
    return round(float(value), 6)


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
