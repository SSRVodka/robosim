"""Read and realize ``scene-export/v9-vsim-articulated-resources`` packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from robosim.core.csd import (
    CsdRealizationManifest,
    make_csd_realization_cache_key,
)

_FORMAT = "scene-export/v9-vsim-articulated-resources"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID = re.compile(r'assetserver:asset:id\s*=\s*"([^"]+)"')
_MODE = re.compile(r'assetserver:simulationSupport:physicsMode\s*=\s*"([^"]+)"')
_DIGEST = re.compile(r'assetserver:simulationSupport:resourceDigest\s*=\s*"sha256:([0-9a-f]{64})"')
_OBJ = re.compile(r"@([^@]+\.obj)@")
_ROBOT_ID = re.compile(r'robosim:robot:id\s*=\s*"([^"]+)"')
_ROBOT_INSTANCE_ID = re.compile(r'robosim:robot:instanceId\s*=\s*"([^"]+)"')


@dataclass(frozen=True, slots=True)
class OpenUsdRigidBody:
    path: str
    visual_obj: Path
    collision_objs: tuple[Path, ...]
    mass: float
    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OpenUsdArticulationJoint:
    name: str
    kind: str
    parent: str
    child: str
    axis: str
    parent_pos: tuple[float, float, float]
    parent_quat: tuple[float, float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    limit: tuple[float, float]
    stiffness: float
    damping: float
    target: float


@dataclass(frozen=True, slots=True)
class OpenUsdVisualMaterial:
    name: str
    texture: Path | None
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OpenUsdAsset:
    asset_id: str
    resource_digest: str
    source: Path
    mode: str
    bodies: tuple[OpenUsdRigidBody, ...]
    joints: tuple[OpenUsdArticulationJoint, ...]
    visual_materials: tuple[OpenUsdVisualMaterial, ...]
    dynamic_friction: float
    static_friction: float
    restitution: float


@dataclass(frozen=True, slots=True)
class OpenUsdSceneInstance:
    prim_path: str
    asset: OpenUsdAsset
    pose: tuple[float, float, float, float, float, float, float]
    joint_targets: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class OpenUsdRobot:
    """A fixed-base robot described by a v9 scene prim."""

    robot_id: str
    instance_id: str
    pose: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class OpenUsdCamera:
    """A scene-authored perspective camera."""

    name: str
    pose: tuple[float, float, float, float, float, float, float]
    fovy: float


@dataclass(frozen=True, slots=True)
class OpenUsdDistantLight:
    """A scene-authored directional light."""

    name: str
    pose: tuple[float, float, float, float, float, float, float]
    intensity: float


@dataclass(frozen=True, slots=True)
class OpenUsdScenePackage:
    root: Path
    scene_id: str
    scene: Path
    instances: tuple[OpenUsdSceneInstance, ...]
    robot: OpenUsdRobot | None
    cameras: tuple[OpenUsdCamera, ...]
    lights: tuple[OpenUsdDistantLight, ...]


class PackageError(ValueError):
    """A package validation error which must become a typed blocker."""


def read_openusd_scene_package(scene_path: Path) -> OpenUsdScenePackage:
    """Read the v9 resource view from the composed OpenUSD stage."""
    from pxr import Usd, UsdGeom

    scene = scene_path.resolve()
    root = scene.parent
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version", manifest.get("format")) != _FORMAT:
        raise PackageError("manifest schema is not scene-export/v9-vsim-articulated-resources")
    if manifest.get("entrypoint") != scene.name:
        raise PackageError("manifest entrypoint does not name the supplied scene")
    _verify_checksums(root)
    stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadAll)
    if stage is None:
        raise PackageError(f"cannot compose scene USD: {scene}")
    if stage.GetMetadata("metersPerUnit") != 1 or stage.GetMetadata("upAxis") != "Z":
        raise PackageError("scene must use metersPerUnit=1 and upAxis=Z")
    instances: list[OpenUsdSceneInstance] = []
    seen: dict[str, tuple[Path, str]] = {}
    for category in ("Rooms", "Objects"):
        category_prim = stage.GetPrimAtPath(f"/World/{category}")
        for prim in category_prim.GetChildren():
            if not prim.IsA(UsdGeom.Xform):
                raise PackageError(f"{prim.GetPath()} is not an Xform instance")
            pose = _stage_pose(prim)
            asset_path = _asset_source(root, prim)
            asset = _read_asset(asset_path)
            name = prim.GetName()
            if category == "Rooms" and asset.mode != "static":
                raise PackageError(f"room {name} is not static")
            if category == "Objects" and asset.mode not in {"rigid", "articulated"}:
                raise PackageError(f"object {name} has unsupported physics mode {asset.mode}")
            previous = seen.get(asset.asset_id)
            identity = (asset.source, asset.resource_digest)
            if previous is not None and previous != identity:
                raise PackageError(f"asset identity {asset.asset_id} resolves inconsistently")
            seen[asset.asset_id] = identity
            instances.append(
                OpenUsdSceneInstance(
                    str(prim.GetPath()), asset, pose, _instance_joint_targets(prim, asset)
                )
            )
    if not instances:
        raise PackageError("scene has no direct room/object references")
    scene_id = str(manifest.get("scene_id", ""))
    if not scene_id:
        raise PackageError("manifest has no scene_id")
    return OpenUsdScenePackage(
        root,
        scene_id,
        scene,
        tuple(instances),
        _read_stage_robot(stage),
        _read_stage_cameras(stage),
        _read_stage_lights(stage),
    )


def compile_openusd_scene_package(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None,
    realization_version: str,
    simulator_version: str | None,
) -> CsdRealizationManifest:
    package = read_openusd_scene_package(csd_path)
    config = dict(realization_config or {})
    closure = _dependency_hash(package.root)
    resources = {asset.asset_id: asset.resource_digest for asset in _unique_assets(package)}
    robot_template = _robot_template(package.robot) if package.robot is not None else None
    if package.robot is not None and robot_template is not None:
        resources[f"robot:{package.robot.robot_id}"] = _directory_hash(robot_template)
    cache = make_csd_realization_cache_key(
        csd_hash=closure,
        asset_variant_hashes=resources,
        backend="mujoco",
        realization_config=config,
        realization_version=f"{realization_version}-mujoco-openusd-0.4",
        simulator_version=simulator_version,
    )
    root = (Path(output_root) / "mujoco" / package.scene_id).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        cached = CsdRealizationManifest.from_json_dict(_json(manifest_path))
        if cached.cache_key == cache.digest and all(
            (root / item).is_file() for item in cached.generated_files
        ):
            return cached
    diagnostics = root / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(exist_ok=True)
    assets = _unique_assets(package)
    model_names: dict[tuple[str, str], str] = {}
    generated: list[str] = ["manifest.json", "scene.xml"]
    mappings: dict[str, Any] = {}
    for asset in assets:
        key = _asset_key(asset)
        model_names[(asset.asset_id, asset.resource_digest)] = key
        model_root = root / "models" / key
        _copy_support(asset.source.parent, model_root)
        generated.extend(_copy_materials(asset, model_root))
        _write_asset(model_root / "asset.xml", asset)
        _load_asset(model_root / "asset.xml")
        report = f"diagnostics/{key}-load.json"
        (root / report).write_text(
            json.dumps({"asset_id": asset.asset_id, "status": "passed"}, indent=2)
        )
        generated.extend((f"models/{key}/asset.xml", report))
    robot_include, robot_files = _copy_robot(
        root=root,
        robot=package.robot,
        template=robot_template,
    )
    generated.extend(robot_files)
    _write_scene(root / "scene.xml", package, model_names, mappings, robot_include)
    initial_state_file = "runtime/initial_joint_positions.json"
    initial_positions = _initial_joint_positions(package)
    runtime = root / initial_state_file
    runtime.parent.mkdir(exist_ok=True)
    runtime.write_text(json.dumps(initial_positions, indent=2, sort_keys=True))
    _validate_scene(root / "scene.xml", diagnostics, initial_positions)
    preview_file = _write_preview(root / "scene.xml", diagnostics, initial_positions)
    generated.extend(
        (
            "diagnostics/scene-load.json",
            "diagnostics/physics.json",
            "diagnostics/entity_mapping.json",
            initial_state_file,
            preview_file,
        )
    )
    manifest = CsdRealizationManifest(
        manifest_id=f"manifest_mujoco_{package.scene_id}",
        csd_id=package.scene_id,
        backend="mujoco",
        cache_key=cache.digest,
        root_path=str(root),
        entry_file="scene.xml",
        generated_files=tuple(generated),
        preview_files=(preview_file,),
        initial_state_file=initial_state_file,
    )
    (diagnostics / "entity_mapping.json").write_text(json.dumps(mappings, indent=2, sort_keys=True))
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True)
    )
    return manifest


def _scene_references(text: str) -> list[tuple[str, str, str, str]]:
    result: list[tuple[str, str, str, str]] = []
    for category in ("Rooms", "Objects"):
        scope = _named_block(text, 'def Xform "' + category + '"')
        for match in re.finditer(
            r'def Xform "([^"]+)"\s*\(\s*prepend references = @([^@]+/asset\.usda)@', scope
        ):
            start = match.start()
            result.append((category, match.group(1), match.group(2), _balanced(scope, start)))
    return result


def _read_robot(text: str) -> OpenUsdRobot | None:
    scope = _named_block(text, 'def Xform "Robot"')
    if not scope:
        return None
    robot_id = _one(_ROBOT_ID, scope, "robot id")
    instance_id = _one(_ROBOT_INSTANCE_ID, scope, "robot instance id")
    if not _unit_scale(scope):
        raise PackageError("robot has non-unit instance scale")
    return OpenUsdRobot(robot_id, instance_id, _pose(scope))


def _read_cameras(text: str) -> tuple[OpenUsdCamera, ...]:
    scope = _named_block(text, 'def Scope "Cameras"')
    result: list[OpenUsdCamera] = []
    for match in re.finditer(r'def Camera "([^"]+)"', scope):
        block = _balanced(scope, match.start())
        focal_length = _number(block, "focalLength", default=50.0)
        aperture = _number(block, "verticalAperture", default=15.0)
        result.append(
            OpenUsdCamera(
                match.group(1),
                _pose(block),
                math.degrees(2 * math.atan(aperture / (2 * focal_length))),
            )
        )
    return tuple(result)


def _read_lights(text: str) -> tuple[OpenUsdDistantLight, ...]:
    scope = _named_block(text, 'def Scope "Lights"')
    result: list[OpenUsdDistantLight] = []
    for match in re.finditer(r'def DistantLight "([^"]+)"', scope):
        block = _balanced(scope, match.start())
        result.append(
            OpenUsdDistantLight(
                match.group(1),
                _pose(block),
                _number(block, "intensity", default=1.0),
            )
        )
    return tuple(result)


def _robot_template(robot: OpenUsdRobot) -> Path:
    if robot.robot_id != "franka_panda":
        raise PackageError(f"unsupported MuJoCo robot: {robot.robot_id}")
    path = (
        Path(__file__).resolve().parents[2]
        / "drivers_sim"
        / "mujoco"
        / "assets"
        / "robots"
        / "franka_panda"
    )
    if not (path / "panda.xml").is_file() or not (path / "panda.srdf").is_file():
        raise PackageError(f"MuJoCo robot template is incomplete: {path}")
    return path


def _copy_robot(
    *,
    root: Path,
    robot: OpenUsdRobot | None,
    template: Path | None,
) -> tuple[str | None, tuple[str, ...]]:
    if robot is None or template is None:
        return None, ()
    destination = root / "robots" / robot.robot_id
    shutil.copytree(template, destination, dirs_exist_ok=True)
    entry = destination / "panda.xml"
    xml = ET.parse(entry)
    body = xml.getroot().find("worldbody/body")
    if body is None:
        raise PackageError(f"robot template has no root body: {entry}")
    body.set("pos", _values(robot.pose[:3]))
    body.set("quat", _values(robot.pose[3:]))
    world = xml.getroot().find("worldbody")
    if world is not None:
        for light in world.findall("light"):
            world.remove(light)
    for parent in xml.getroot().iter():
        for camera in parent.findall("camera"):
            parent.remove(camera)
    compiler = xml.getroot().find("compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
    for mesh in xml.getroot().findall("asset/mesh"):
        file = mesh.get("file")
        if file is not None and "/" not in file:
            mesh.set("file", f"assets/{file}")
    ET.indent(xml)
    xml.write(entry, encoding="unicode", xml_declaration=False)
    files = tuple(str(path.relative_to(root)) for path in destination.rglob("*") if path.is_file())
    return str(Path("robots") / robot.robot_id / "panda.xml"), files


def _read_asset(path: Path) -> OpenUsdAsset:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise PackageError(f"cannot compose asset USD: {path}")
    root = stage.GetDefaultPrim()
    if not root:
        raise PackageError(f"asset has no default prim: {path}")
    if "shells" in path.parts and root.GetAttribute("assetserver:asset:id").Get() is None:
        return _read_shell_asset(path)
    asset_id = _attribute(root, "assetserver:asset:id", str)
    digest = _attribute(root, "assetserver:simulationSupport:resourceDigest", str)
    mode = _attribute(root, "assetserver:simulationSupport:physicsMode", str)
    if not digest.startswith("sha256:") or not _SHA256.fullmatch(digest[7:]):
        raise PackageError(f"{asset_id} has invalid resourceDigest")
    bodies: list[OpenUsdRigidBody] = []
    body_prims = (
        tuple(
            prim
            for prim in stage.GetPrimAtPath("/Asset/Links").GetChildren()
            if "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas()
        )
        if mode == "articulated"
        else (root,)
    )
    for body in body_prims:
        name = body.GetName() if mode == "articulated" else "asset_root"
        visual_path = _asset_path(body, "assetserver:simulationSupport:visualObj")
        collision = _asset_paths(body, "assetserver:simulationSupport:collisionObjs")
        if visual_path is None or not collision:
            raise PackageError(f"{asset_id}:{name} lacks visual or collision OBJ")
        visual = Path(visual_path)
        collision_paths = tuple(Path(item) for item in collision)
        if visual.suffix != ".obj" or any(item.suffix != ".obj" for item in collision_paths):
            raise PackageError(f"{asset_id}:{name} has unsupported mesh format")
        if mode == "static":
            bodies.append(
                OpenUsdRigidBody(
                    name,
                    visual,
                    collision_paths,
                    0.0,
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0, 0.0),
                )
            )
        else:
            bodies.append(
                OpenUsdRigidBody(
                    name,
                    visual,
                    collision_paths,
                    _attribute(body, "physics:mass", float),
                    cast(
                        tuple[float, float, float],
                        _vector_attribute(body, "physics:centerOfMass", 3),
                    ),
                    cast(
                        tuple[float, float, float],
                        _vector_attribute(body, "physics:diagonalInertia", 3),
                    ),
                    cast(
                        tuple[float, float, float, float],
                        _vector_attribute(body, "physics:principalAxes", 4),
                    ),
                )
            )
    joints = _read_stage_joints(stage) if mode == "articulated" else ()
    if mode == "articulated":
        _validate_tree(bodies, joints)
    materials = _read_visual_materials(path, tuple(body.visual_obj for body in bodies))
    dynamic_friction, static_friction, restitution = _physics_material(stage, root)
    return OpenUsdAsset(
        asset_id,
        digest[7:],
        path.resolve(),
        mode,
        tuple(bodies),
        tuple(joints),
        materials,
        dynamic_friction,
        static_friction,
        restitution,
    )


def _read_shell_asset(path: Path) -> OpenUsdAsset:
    """Read the v9d procedural shell export with separate floor/wall resources."""
    digest = _directory_hash(path.parent)
    support = path.parent / "support" / "obj"
    bodies = (
        OpenUsdRigidBody(
            "Floor",
            Path("support/obj/floor/visual.obj"),
            (Path("support/obj/floor/collision_000.obj"),),
            0.0,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        OpenUsdRigidBody(
            "Walls",
            Path("support/obj/walls/visual.obj"),
            tuple(Path(f"support/obj/walls/collision_{index:03d}.obj") for index in range(4)),
            0.0,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
    )
    if any(
        not (path.parent / item).is_file()
        for body in bodies
        for item in (body.visual_obj, *body.collision_objs)
    ):
        raise PackageError(f"shell support closure is incomplete: {support}")
    materials = _read_shell_materials(path)
    return OpenUsdAsset(
        f"shell://sha256/{digest}",
        digest,
        path.resolve(),
        "static",
        bodies,
        (),
        materials,
        0.5,
        0.5,
        0.0,
    )


def _attribute(prim: Any, name: str, expected: type[str] | type[float]) -> Any:
    value = prim.GetAttribute(name).Get()
    if value is None or not isinstance(value, expected):
        raise PackageError(f"{prim.GetPath()} missing {name}")
    return value


def _vector_attribute(prim: Any, name: str, size: int) -> tuple[float, ...]:
    value = prim.GetAttribute(name).Get()
    if size == 4 and value is not None and hasattr(value, "GetReal"):
        imaginary = value.GetImaginary()
        return (float(value.GetReal()), *(float(item) for item in imaginary))
    if value is None or len(value) != size:
        raise PackageError(f"{prim.GetPath()} missing {name}")
    return tuple(float(item) for item in value)


def _asset_path(prim: Any, name: str) -> str | None:
    value = prim.GetAttribute(name).Get()
    return value.path if value is not None else None


def _asset_paths(prim: Any, name: str) -> tuple[str, ...]:
    value = prim.GetAttribute(name).Get()
    return tuple(item.path for item in value) if value is not None else ()


def _physics_material(stage: Any, root: Any) -> tuple[float, float, float]:
    """Read the required asset-level PhysicsMaterialAPI values."""
    material = stage.GetPrimAtPath(root.GetPath().AppendChild("PhysicsMaterial"))
    if not material or "PhysicsMaterialAPI" not in material.GetAppliedSchemas():
        raise PackageError(f"{root.GetPath()} has no PhysicsMaterialAPI")
    dynamic = _attribute(material, "physics:dynamicFriction", float)
    static = _attribute(material, "physics:staticFriction", float)
    restitution = _attribute(material, "physics:restitution", float)
    if dynamic < 0 or static < 0 or not 0 <= restitution <= 1:
        raise PackageError(f"{material.GetPath()} has invalid contact material")
    return dynamic, static, restitution


def _read_shell_materials(path: Path) -> tuple[OpenUsdVisualMaterial, ...]:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise PackageError(f"cannot compose shell USD: {path}")
    materials: list[OpenUsdVisualMaterial] = []
    for name in ("Floor", "Walls"):
        prim = stage.GetPrimAtPath(f"/Asset/{name}")
        if not prim:
            prim = stage.GetPrimAtPath(f"/Asset/Visual/{name}")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise PackageError(f"shell visual {name} has no bound material")
        texture = next(
            (
                shader.GetInput("file").Get().resolvedPath
                for child in material.GetPrim().GetChildren()
                if child.GetTypeName() == "Shader"
                and (shader := UsdShade.Shader(child)).GetInput("file").Get()
                and child.GetName() == "ColorTexture"
            ),
            "",
        )
        if not texture:
            raise PackageError(f"shell visual {name} has no color texture")
        texture_path = Path(texture).resolve()
        if not texture_path.is_file():
            raise PackageError(f"shell visual {name} texture is missing: {texture}")
        materials.append(OpenUsdVisualMaterial(name.lower(), texture_path, (1.0, 1.0, 1.0, 1.0)))
    return tuple(materials)


def _read_visual_materials(
    path: Path, visuals: tuple[Path, ...]
) -> tuple[OpenUsdVisualMaterial, ...]:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise PackageError(f"cannot compose asset USD: {path}")
    requested = {
        name
        for visual in visuals
        for name in _obj_material_names(path.parent / visual)
        if name is not None
    }
    if not requested:
        return ()
    available: dict[str, tuple[Path | None, tuple[float, float, float, float]]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material" or "PhysicsMaterialAPI" in prim.GetAppliedSchemas():
            continue
        surface = UsdShade.Material(prim).ComputeSurfaceSource()[0]
        if not surface:
            continue
        shader = UsdShade.Shader(surface)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        diffuse = shader.GetInput("diffuseColor").Get() or (0.8, 0.8, 0.8)
        opacity = float(shader.GetInput("opacity").Get() or 1.0)
        texture = next(
            (
                Path(input_.Get().resolvedPath)
                for child in prim.GetChildren()
                if (candidate := UsdShade.Shader(child)).GetIdAttr().Get() == "UsdUVTexture"
                for input_ in (candidate.GetInput("file"),)
                if input_.Get() and input_.Get().resolvedPath
            ),
            None,
        )
        value = (texture, (float(diffuse[0]), float(diffuse[1]), float(diffuse[2]), opacity))
        key = str(prim.GetPath()).lstrip("/").replace("/", "_")
        available[key] = value
        if key.startswith("Asset_"):
            available[key.removeprefix("Asset_")] = value
    missing = requested - available.keys()
    if missing:
        raise PackageError(
            f"{path} has OBJ materials without USD PreviewSurface: {sorted(missing)}"
        )
    return tuple(OpenUsdVisualMaterial(name, *available[name]) for name in sorted(requested))


def _read_joints(text: str) -> tuple[OpenUsdArticulationJoint, ...]:
    result: list[OpenUsdArticulationJoint] = []
    for match in re.finditer(r'def Physics(Revolute|Prismatic)Joint "([^"]+)"', text):
        block = _balanced(text, match.start())
        parent = _relationship(block, "physics:body0")
        child = _relationship(block, "physics:body1")
        kind = match.group(1).lower()
        limit = (_number(block, "physics:lowerLimit"), _number(block, "physics:upperLimit"))
        if kind == "revolute":
            limit = (math.radians(limit[0]), math.radians(limit[1]))
        result.append(
            OpenUsdArticulationJoint(
                match.group(2),
                kind,
                parent.rsplit("/", 1)[-1],
                child.rsplit("/", 1)[-1],
                _one(re.compile(r'physics:axis\s*=\s*"([XYZ])"'), block, "joint axis"),
                _triple(block, "physics:localPos0"),
                _quat(block, "physics:localRot0"),
                _triple(block, "physics:localPos1"),
                _quat(block, "physics:localRot1"),
                limit,
                _number(block, "physics:stiffness", default=0.0),
                _number(block, "physics:damping", default=0.0),
                _number(block, "physics:targetPosition", default=0.0),
            )
        )
    return tuple(result)


def _read_stage_joints(stage: Any) -> tuple[OpenUsdArticulationJoint, ...]:
    result: list[OpenUsdArticulationJoint] = []
    for prim in stage.GetPrimAtPath("/Asset/Joints").GetChildren():
        kind = {"PhysicsRevoluteJoint": "revolute", "PhysicsPrismaticJoint": "prismatic"}.get(
            prim.GetTypeName()
        )
        if kind is None:
            raise PackageError(f"unsupported articulation joint: {prim.GetPath()}")
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise PackageError(f"{prim.GetPath()} must have one body0 and body1")
        limit = (
            float(_attribute(prim, "physics:lowerLimit", float)),
            float(_attribute(prim, "physics:upperLimit", float)),
        )
        if kind == "revolute":
            limit = (math.radians(limit[0]), math.radians(limit[1]))
        result.append(
            OpenUsdArticulationJoint(
                prim.GetName(),
                kind,
                body0[0].name,
                body1[0].name,
                _attribute(prim, "physics:axis", str),
                cast(tuple[float, float, float], _vector_attribute(prim, "physics:localPos0", 3)),
                cast(
                    tuple[float, float, float, float],
                    _vector_attribute(prim, "physics:localRot0", 4),
                ),
                cast(tuple[float, float, float], _vector_attribute(prim, "physics:localPos1", 3)),
                cast(
                    tuple[float, float, float, float],
                    _vector_attribute(prim, "physics:localRot1", 4),
                ),
                limit,
                float(
                    prim.GetAttribute("drive:angular:physics:stiffness").Get()
                    or prim.GetAttribute("drive:linear:physics:stiffness").Get()
                    or 0
                ),
                float(
                    prim.GetAttribute("drive:angular:physics:damping").Get()
                    or prim.GetAttribute("drive:linear:physics:damping").Get()
                    or 0
                ),
                0.0,
            )
        )
    return tuple(result)


def _asset_source(root: Path, prim: Any) -> Path:
    stack = prim.GetPrimStack()
    if len(stack) < 2:
        raise PackageError(f"{prim.GetPath()} has no asset reference")
    source = _inside(root, Path(stack[1].layer.realPath))
    if source.name != "asset.usda":
        raise PackageError(f"{prim.GetPath()} reference is not asset.usda")
    return source


def _stage_pose(prim: Any) -> tuple[float, float, float, float, float, float, float]:
    from pxr import UsdGeom

    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    rows = [matrix.GetRow3(index) for index in range(3)]
    if any(abs(float(row.GetLength()) - 1.0) > 1e-6 for row in rows):
        raise PackageError(f"{prim.GetPath()} has non-unit instance scale")
    if any(
        abs(float(rows[left] * rows[right])) > 1e-6 for left in range(3) for right in range(left)
    ):
        raise PackageError(f"{prim.GetPath()} has shear or an unsupported transform")
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotation()
    quat = rotation.GetQuat()
    imaginary = quat.GetImaginary()
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(quat.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _instance_joint_targets(instance: Any, asset: OpenUsdAsset) -> tuple[tuple[str, float], ...]:
    targets: list[tuple[str, float]] = []
    for joint in asset.joints:
        prim = instance.GetStage().GetPrimAtPath(
            instance.GetPath().AppendPath("Joints").AppendChild(joint.name)
        )
        drive = "angular" if joint.kind == "revolute" else "linear"
        value = prim.GetAttribute(f"drive:{drive}:physics:targetPosition").Get()
        if value is None:
            value = 0.0
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PackageError(f"{prim.GetPath()} has invalid targetPosition")
        target = math.radians(float(value)) if joint.kind == "revolute" else float(value)
        if target < joint.limit[0] or target > joint.limit[1]:
            raise PackageError(f"{prim.GetPath()} targetPosition is outside joint limits")
        targets.append((joint.name, target))
    return tuple(targets)


def _read_stage_robot(stage: Any) -> OpenUsdRobot | None:
    prim = stage.GetPrimAtPath("/World/Robot")
    if not prim:
        return None
    return OpenUsdRobot(
        _attribute(prim, "robosim:robot:id", str),
        _attribute(prim, "robosim:robot:instanceId", str),
        _stage_pose(prim),
    )


def _read_stage_cameras(stage: Any) -> tuple[OpenUsdCamera, ...]:
    cameras = stage.GetPrimAtPath("/World/Cameras")
    if not cameras:
        return ()
    result: list[OpenUsdCamera] = []
    for prim in cameras.GetChildren():
        if prim.GetTypeName() != "Camera":
            raise PackageError(f"unsupported camera schema: {prim.GetPath()}")
        focal = float(prim.GetAttribute("focalLength").Get() or 50.0)
        aperture = float(prim.GetAttribute("verticalAperture").Get() or 15.0)
        result.append(
            OpenUsdCamera(
                prim.GetName(),
                _stage_pose(prim),
                math.degrees(2 * math.atan(aperture / (2 * focal))),
            )
        )
    return tuple(result)


def _read_stage_lights(stage: Any) -> tuple[OpenUsdDistantLight, ...]:
    lights = stage.GetPrimAtPath("/World/Lights")
    if not lights:
        return ()
    result: list[OpenUsdDistantLight] = []
    for prim in lights.GetChildren():
        if prim.GetTypeName() != "DistantLight":
            raise PackageError(f"unsupported light schema: {prim.GetPath()}")
        result.append(
            OpenUsdDistantLight(
                prim.GetName(),
                _stage_pose(prim),
                float(prim.GetAttribute("intensity").Get() or 1.0),
            )
        )
    return tuple(result)


def _write_asset(path: Path, asset: OpenUsdAsset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": _asset_key(asset)})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "meshdir": "support/obj", "texturedir": "textures"},
    )
    assets = ET.SubElement(root, "asset")
    materials = {item.name: f"{item.name}_material" for item in asset.visual_materials}
    for item in asset.visual_materials:
        attrs = {"name": materials[item.name], "rgba": _values(item.rgba)}
        if item.texture is not None:
            texture = f"{item.name}_texture"
            ET.SubElement(
                assets,
                "texture",
                {"name": texture, "type": "2d", "file": item.texture.with_suffix(".png").name},
            )
            attrs["texture"] = texture
        ET.SubElement(assets, "material", attrs)
    world = ET.SubElement(root, "worldbody")
    bodies = {body.path: body for body in asset.bodies}
    children = {joint.child: joint for joint in asset.joints}
    roots = [body for body in asset.bodies if body.path not in children]
    if asset.mode != "static" and len(roots) != 1:
        raise PackageError(f"{asset.asset_id} does not have one articulation root")

    def emit(
        parent: ET.Element, body: OpenUsdRigidBody, joint: OpenUsdArticulationJoint | None
    ) -> None:
        attrs = {
            "name": (
                "asset_root"
                if parent is world
                else "static_root"
                if asset.mode == "static" and body.path == "asset_root"
                else body.path
            )
        }
        if joint is not None:
            pos, quat = _parent_to_child(joint)
            attrs.update({"pos": _values(pos), "quat": _values(quat)})
        node = ET.SubElement(parent, "body", attrs)
        if parent is world and asset.mode in {"rigid", "articulated"}:
            ET.SubElement(node, "freejoint", {"name": "freejoint"})
        if asset.mode != "static":
            ET.SubElement(
                node,
                "inertial",
                {
                    "pos": _values(body.center_of_mass),
                    "quat": _values(body.principal_axes),
                    "mass": str(body.mass),
                    "diaginertia": _values(body.diagonal_inertia),
                },
            )
        visual_objs = (
            _shell_visual_objs(path, body.visual_obj, asset.visual_materials)
            if asset.mode == "static"
            else _visual_obj_parts(path, body.visual_obj)
        )
        parts = [(item, name, True) for item, name in visual_objs]
        parts.extend((item, None, False) for item in body.collision_objs)
        for index, (obj, material, is_visual) in enumerate(parts):
            mesh = f"{body.path}_{index}"
            flat_box = (
                _flat_obj(path.parent / obj)
                if not is_visual or obj.name != body.visual_obj.name
                else None
            )
            geom = {"name": mesh, "density": "0"}
            if flat_box is None:
                ET.SubElement(
                    assets,
                    "mesh",
                    {"name": mesh, "file": obj.relative_to("support/obj").as_posix()},
                )
                geom.update({"type": "mesh", "mesh": mesh})
            else:
                geom.update(
                    {"type": "box", "pos": _values(flat_box[0]), "size": _values(flat_box[1])}
                )
            if is_visual:
                geom.update({"contype": "0", "conaffinity": "0"})
            else:
                geom["rgba"] = "0 0 0 0"
            if material is not None:
                geom["material"] = materials[material]
            ET.SubElement(node, "geom", geom)
        for child in (item for item in asset.joints if item.parent == body.path):
            emit(node, bodies[child.child], child)
        if joint is not None:
            attrs = {
                "name": joint.name,
                "type": "hinge" if joint.kind == "revolute" else "slide",
                "pos": _values(joint.pos),
                "axis": _axis(joint.axis, joint.quat),
                "range": _values(joint.limit),
                "limited": "true",
                "stiffness": str(joint.stiffness),
                "damping": str(joint.damping),
                "springref": str(
                    math.radians(joint.target) if joint.kind == "revolute" else joint.target
                ),
            }
            ET.SubElement(node, "joint", attrs)

    if asset.mode == "static":
        container = ET.SubElement(world, "body", {"name": "asset_root"})
        for body in roots:
            emit(container, body, None)
    else:
        emit(world, roots[0], None)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _write_scene(
    path: Path,
    package: OpenUsdScenePackage,
    models: Mapping[tuple[str, str], str],
    mappings: dict[str, Any],
    robot_include: str | None,
) -> None:
    root = ET.Element("mujoco", {"model": package.scene_id})
    if robot_include is not None:
        ET.SubElement(root, "include", {"file": robot_include})
        robot = package.robot
        if robot is not None:
            mappings["/World/Robot"] = {
                "robot_id": robot.robot_id,
                "instance_id": robot.instance_id,
            }
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "512", "offheight": "512"})
    assets = ET.SubElement(root, "asset")
    for model in sorted(set(models.values())):
        ET.SubElement(assets, "model", {"name": model, "file": f"models/{model}/asset.xml"})
    world = ET.SubElement(root, "worldbody")
    for instance in package.instances:
        model = models[(instance.asset.asset_id, instance.asset.resource_digest)]
        prefix = _prefix(instance.prim_path)
        frame = ET.SubElement(
            world,
            "frame",
            {"name": prefix, "pos": _values(instance.pose[:3]), "quat": _values(instance.pose[3:])},
        )
        ET.SubElement(
            frame, "attach", {"model": model, "body": "asset_root", "prefix": prefix + "/"}
        )
        mappings[instance.prim_path] = {"frame": prefix, "prefix": prefix + "/"}
    if not package.cameras:
        ET.SubElement(
            world,
            "camera",
            {
                "name": "world_camera",
                "pos": "4 -4 3",
                "xyaxes": "0.707 0.707 0 -0.408 0.408 0.816",
            },
        )
    for camera in package.cameras:
        orientation = _normalize(camera.pose[3:])
        ET.SubElement(
            world,
            "camera",
            {
                "name": camera.name,
                "pos": _values(camera.pose[:3]),
                "xyaxes": _values(
                    (*_rotate(orientation, (1.0, 0.0, 0.0)), *_rotate(orientation, (0.0, 1.0, 0.0)))
                ),
                "fovy": str(camera.fovy),
            },
        )
        mappings[f"/World/Cameras/{camera.name}"] = {"camera": camera.name}
    for light in package.lights:
        ET.SubElement(
            world,
            "light",
            {
                "name": light.name,
                "directional": "true",
                "pos": _values(light.pose[:3]),
                "dir": _values(_rotate(_normalize(light.pose[3:]), (0.0, 0.0, -1.0))),
                "diffuse": _values((light.intensity, light.intensity, light.intensity)),
            },
        )
        mappings[f"/World/Lights/{light.name}"] = {"light": light.name}
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _load_asset(path: Path) -> None:
    import mujoco

    mujoco.MjModel.from_xml_path(str(path))


def _initial_joint_positions(package: OpenUsdScenePackage) -> dict[str, float]:
    return {
        f"{_prefix(instance.prim_path)}/{joint}": target
        for instance in package.instances
        for joint, target in instance.joint_targets
    }


def _apply_initial_positions(model: Any, data: Any, positions: Mapping[str, float]) -> None:
    import mujoco

    for name, value in positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or int(model.jnt_type[joint_id]) not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            raise PackageError(f"initial state names unsupported joint: {name}")
        if not math.isfinite(value):
            raise PackageError(f"initial state is non-finite: {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value


def _validate_scene(path: Path, diagnostics: Path, initial_positions: Mapping[str, float]) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    _apply_initial_positions(model, data, initial_positions)
    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
    finite = bool(all(math.isfinite(float(v)) for v in (*data.qpos, *data.qvel)))
    if not finite:
        raise PackageError("MuJoCo stepping produced non-finite state")
    (diagnostics / "scene-load.json").write_text(
        json.dumps({"status": "passed", "nbody": model.nbody}, indent=2)
    )
    (diagnostics / "physics.json").write_text(
        json.dumps({"status": "passed", "steps": 100}, indent=2)
    )


def _write_preview(
    scene_path: Path, diagnostics: Path, initial_positions: Mapping[str, float]
) -> str:
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    _apply_initial_positions(model, data, initial_positions)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        renderer.update_scene(data, camera=0)
        image = renderer.render()
    filename = "diagnostics/preview.png"
    Image.fromarray(image).save(diagnostics.parent / filename)
    return filename


def _verify_checksums(root: Path) -> None:
    checksums = root / "checksums.sha256"
    if not checksums.is_file():
        raise PackageError("checksums.sha256 is missing")
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        if not _SHA256.fullmatch(digest):
            raise PackageError("invalid checksum digest")
        candidate = _inside(root, root / relative.strip())
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise PackageError(f"checksum mismatch: {relative}")


def _copy_support(source: Path, destination: Path) -> None:
    support = source / "support"
    if not support.is_dir():
        raise PackageError(f"{source} has no support directory")
    shutil.copytree(support, destination / "support", dirs_exist_ok=True)


def _copy_materials(asset: OpenUsdAsset, destination: Path) -> list[str]:
    from PIL import Image

    generated: list[str] = []
    textures = destination / "textures"
    textures.mkdir(exist_ok=True)
    for material in asset.visual_materials:
        if material.texture is None:
            continue
        target = textures / material.texture.with_suffix(".png").name
        with Image.open(material.texture) as image:
            image.convert("RGB").save(target)
        generated.append(target.relative_to(destination.parent.parent).as_posix())
    return generated


def _shell_visual_objs(
    asset_xml: Path,
    visual: Path,
    materials: tuple[OpenUsdVisualMaterial, ...],
) -> tuple[tuple[Path, str | None], ...]:
    if not materials:
        return ((visual, None),)
    for material in materials:
        if visual == Path("support/obj") / material.name / "visual.obj":
            return tuple((item, material.name) for item in _split_visual_obj(asset_xml, visual))
    source = asset_xml.parent / visual
    lines = source.read_text().splitlines()
    prefix = [line for line in lines if not line.startswith(("o ", "usemtl ", "f "))]
    result: list[tuple[Path, str]] = []
    for material in materials:
        selected: list[str] = []
        active = False
        for line in lines:
            if line.startswith("o "):
                active = f"_Visual_{material.name.title()}" in line
            elif active and line.startswith("f "):
                selected.append(line)
        if selected:
            target = source.with_name(f"visual_{material.name}.obj")
            target.write_text("\n".join((*prefix, *selected, "")))
            result.append((target.relative_to(asset_xml.parent), material.name))
        elif material.name == "floor":
            result.append((Path("support/obj/collision_000.obj"), material.name))
        else:
            raise PackageError(f"shell visual OBJ has no {material.name} geometry")
    return tuple(result)


def _split_visual_obj(asset_xml: Path, visual: Path) -> tuple[Path, ...]:
    source = asset_xml.parent / visual
    lines = source.read_text().splitlines()
    groups: list[list[str]] = []
    active: list[str] | None = None
    for line in lines:
        if line.startswith("o "):
            active = []
            groups.append(active)
        elif active is not None and line.startswith("f "):
            active.append(line)
    if len(groups) <= 1:
        return (visual,)
    prefix = [line for line in lines if not line.startswith(("o ", "usemtl ", "f "))]
    result: list[Path] = []
    for index, faces in enumerate(groups):
        if not faces:
            continue
        target = source.with_name(f"{source.stem}_{index:03d}.obj")
        target.write_text("\n".join((*prefix, *faces, "")))
        result.append(target.relative_to(asset_xml.parent))
    if not result:
        raise PackageError(f"visual OBJ has no faces: {visual}")
    return tuple(result)


def _visual_obj_parts(asset_xml: Path, visual: Path) -> tuple[tuple[Path, str | None], ...]:
    parts = _split_visual_obj(asset_xml, visual)
    materials = _obj_material_names(asset_xml.parent / visual)
    if len(parts) != len(materials):
        raise PackageError(f"visual OBJ has inconsistent object/material groups: {visual}")
    return tuple(zip(parts, materials, strict=True))


def _obj_material_names(path: Path) -> tuple[str | None, ...]:
    names: list[str | None] = []
    current: str | None = None
    grouped = False
    for line in path.read_text().splitlines():
        if line.startswith("o "):
            if grouped:
                names.append(current)
            grouped = True
            current = None
        elif line.startswith("usemtl "):
            current = line.split(maxsplit=1)[1]
    names.append(current)
    return tuple(names)


def _flat_obj(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    vertices = [
        tuple(float(value) for value in line.split()[1:4])
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    if not vertices:
        raise PackageError(f"OBJ has no vertices: {path}")
    minimum = tuple(min(vertex[index] for vertex in vertices) for index in range(3))
    maximum = tuple(max(vertex[index] for vertex in vertices) for index in range(3))
    size = tuple((maximum[index] - minimum[index]) / 2 for index in range(3))
    if all(value > 1e-8 for value in size):
        return None
    return (
        (minimum[0] + maximum[0]) / 2,
        (minimum[1] + maximum[1]) / 2,
        (minimum[2] + maximum[2]) / 2,
    ), (
        max(size[0], 1e-4),
        max(size[1], 1e-4),
        max(size[2], 1e-4),
    )


def _unique_assets(package: OpenUsdScenePackage) -> tuple[OpenUsdAsset, ...]:
    return tuple(
        {
            (item.asset.asset_id, item.asset.resource_digest): item.asset
            for item in package.instances
        }.values()
    )


def _validate_tree(
    bodies: list[OpenUsdRigidBody], joints: tuple[OpenUsdArticulationJoint, ...]
) -> None:
    names = {body.path for body in bodies}
    children = [joint.child for joint in joints]
    if (
        len(children) != len(set(children))
        or any(j.parent not in names or j.child not in names for j in joints)
        or len(joints) != len(bodies) - 1
    ):
        raise PackageError("articulation is not a single rooted tree")


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise PackageError(f"path escapes package: {path}")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"invalid JSON {path.name}: {error}") from error


def _dependency_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for line in sorted((root / "checksums.sha256").read_text().splitlines()):
        digest.update(line.encode())
    return digest.hexdigest()


def _directory_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _asset_key(asset: OpenUsdAsset) -> str:
    return (
        "asset-"
        + hashlib.sha256((asset.asset_id + asset.resource_digest).encode()).hexdigest()[:16]
    )


def _prefix(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", path.strip("/"))


def _values(values: tuple[float, ...]) -> str:
    return " ".join(str(value) for value in values)


def _one(pattern: re.Pattern[str], text: str, label: str) -> str:
    match = pattern.search(text)
    if not match:
        raise PackageError(f"missing {label}")
    return match.group(1)


def _number(text: str, name: str, default: float | None = None) -> float:
    match = re.search(re.escape(name) + r"\s*=\s*([-+0-9.eE]+)", text)
    if match:
        return float(match.group(1))
    if default is not None:
        return default
    raise PackageError(f"missing {name}")


def _triple(text: str, name: str) -> tuple[float, float, float]:
    return tuple(float(x) for x in _tuple(text, name, 3))  # type: ignore[return-value]


def _quat(text: str, name: str) -> tuple[float, float, float, float]:
    return tuple(float(x) for x in _tuple(text, name, 4))  # type: ignore[return-value]


def _tuple(text: str, name: str, count: int) -> tuple[str, ...]:
    match = re.search(re.escape(name) + r"\s*=\s*\(([^)]+)\)", text)
    if not match:
        raise PackageError(f"missing {name}")
    values = tuple(part.strip() for part in match.group(1).split(","))
    if len(values) != count:
        raise PackageError(f"invalid {name}")
    return values


def _relationship(text: str, name: str) -> str:
    match = re.search(re.escape(name) + r"\s*=\s*<([^>]+)>", text)
    if not match:
        raise PackageError(f"missing {name}")
    return match.group(1)


def _named_block(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    return _balanced(text, start)


def _balanced(text: str, start: int) -> str:
    opening = text.find("{", start)
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise PackageError("unbalanced USDA block")


def _unit_scale(text: str) -> bool:
    return not re.search(r"xformOp:scale\s*=\s*\((?!1\s*,\s*1\s*,\s*1\s*\))", text)


def _pose(text: str) -> tuple[float, float, float, float, float, float, float]:
    pos = _triple(text, "xformOp:translate")
    orient = re.search(r"xformOp:orient\s*=\s*\(([^)]+)\)", text)
    if orient:
        return (*pos, *_quat(text, "xformOp:orient"))
    rotate = _triple(text, "xformOp:rotateXYZ") if "xformOp:rotateXYZ" in text else (0.0, 0.0, 0.0)
    return (*pos, *_euler_quat(rotate))


def _euler_quat(degrees: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = (math.radians(v) / 2 for v in degrees)
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


def _axis(axis: str, quat: tuple[float, float, float, float]) -> str:
    x, y, z = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[axis]
    w, i, j, k = _normalize(quat)
    return _values(
        (
            (1 - 2 * (j * j + k * k)) * x + 2 * (i * j - k * w) * y + 2 * (i * k + j * w) * z,
            2 * (i * j + k * w) * x + (1 - 2 * (i * i + k * k)) * y + 2 * (j * k - i * w) * z,
            2 * (i * k - j * w) * x + 2 * (j * k + i * w) * y + (1 - 2 * (i * i + j * j)) * z,
        )
    )


def _parent_to_child(
    joint: OpenUsdArticulationJoint,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    child_frame = _normalize(joint.quat)
    inverse = (child_frame[0], -child_frame[1], -child_frame[2], -child_frame[3])
    rotation = _multiply(_normalize(joint.parent_quat), inverse)
    rotated = _rotate(rotation, joint.pos)
    position = (
        joint.parent_pos[0] - rotated[0],
        joint.parent_pos[1] - rotated[1],
        joint.parent_pos[2] - rotated[2],
    )
    return position, rotation


def _multiply(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w, x, y, z = left
    a, b, c, d = right
    return (
        w * a - x * b - y * c - z * d,
        w * b + x * a + y * d - z * c,
        w * c - x * d + y * a + z * b,
        w * d + x * c - y * b + z * a,
    )


def _normalize(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(value * value for value in quat))
    return tuple(value / length for value in quat)  # type: ignore[return-value]


def _rotate(
    quat: tuple[float, float, float, float], vector: tuple[float, ...]
) -> tuple[float, float, float]:
    rotated = _multiply(
        _multiply(quat, (0.0, vector[0], vector[1], vector[2])),
        (quat[0], -quat[1], -quat[2], -quat[3]),
    )
    return rotated[1], rotated[2], rotated[3]
