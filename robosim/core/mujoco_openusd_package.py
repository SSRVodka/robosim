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
from typing import Any, Mapping

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
    texture: Path


@dataclass(frozen=True, slots=True)
class OpenUsdAsset:
    asset_id: str
    resource_digest: str
    source: Path
    mode: str
    bodies: tuple[OpenUsdRigidBody, ...]
    joints: tuple[OpenUsdArticulationJoint, ...]
    visual_materials: tuple[OpenUsdVisualMaterial, ...]


@dataclass(frozen=True, slots=True)
class OpenUsdSceneInstance:
    prim_path: str
    asset: OpenUsdAsset
    pose: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class OpenUsdScenePackage:
    root: Path
    scene_id: str
    scene: Path
    instances: tuple[OpenUsdSceneInstance, ...]


class PackageError(ValueError):
    """A package validation error which must become a typed blocker."""


def read_openusd_scene_package(scene_path: Path) -> OpenUsdScenePackage:
    """Read the deliberately small v9 resource view without a registry."""
    scene = scene_path.resolve()
    root = scene.parent
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version", manifest.get("format")) != _FORMAT:
        raise PackageError("manifest schema is not scene-export/v9-vsim-articulated-resources")
    if manifest.get("entrypoint") != scene.name:
        raise PackageError("manifest entrypoint does not name the supplied scene")
    _verify_checksums(root)
    text = scene.read_text()
    if not re.search(r"metersPerUnit\s*=\s*1(?:\D|$)", text) or 'upAxis = "Z"' not in text:
        raise PackageError("scene must use metersPerUnit=1 and upAxis=Z")
    instances: list[OpenUsdSceneInstance] = []
    seen: dict[str, tuple[Path, str]] = {}
    for category, name, source, block in _scene_references(text):
        if not _unit_scale(block):
            raise PackageError(f"{category}/{name} has non-unit instance scale")
        asset_path = _inside(root, root / source)
        asset = _read_asset(asset_path)
        if category == "Rooms" and asset.mode != "static":
            raise PackageError(f"room {name} is not static")
        if category == "Objects" and asset.mode not in {"rigid", "articulated"}:
            raise PackageError(f"object {name} has unsupported physics mode {asset.mode}")
        previous = seen.get(asset.asset_id)
        identity = (asset.source, asset.resource_digest)
        if previous is not None and previous != identity:
            raise PackageError(f"asset identity {asset.asset_id} resolves inconsistently")
        seen[asset.asset_id] = identity
        instances.append(OpenUsdSceneInstance(f"/World/{category}/{name}", asset, _pose(block)))
    if not instances:
        raise PackageError("scene has no direct room/object references")
    scene_id = str(manifest.get("scene_id", ""))
    if not scene_id:
        raise PackageError("manifest has no scene_id")
    return OpenUsdScenePackage(root, scene_id, scene, tuple(instances))


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
    cache = make_csd_realization_cache_key(
        csd_hash=closure,
        asset_variant_hashes=resources,
        backend="mujoco",
        realization_config=config,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )
    root = Path(output_root) / "mujoco" / package.scene_id
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
    _write_scene(root / "scene.xml", package, model_names, mappings)
    _validate_scene(root / "scene.xml", diagnostics)
    preview_file = _write_preview(root / "scene.xml", diagnostics)
    generated.extend(
        (
            "diagnostics/scene-load.json",
            "diagnostics/physics.json",
            "diagnostics/entity_mapping.json",
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


def _read_asset(path: Path) -> OpenUsdAsset:
    text = path.read_text()
    asset_id = _one(_ASSET_ID, text, "asset id")
    digest = _one(_DIGEST, text, "resourceDigest")
    mode = _one(_MODE, text, "physics mode")
    bodies: list[OpenUsdRigidBody] = []
    links = _named_block(text, 'def Scope "Links"') if mode == "articulated" else text
    body_names = re.findall(
        r'(?:def|over) "([^"]+)"\s*\(\s*prepend apiSchemas = \["PhysicsRigidBodyAPI"', links
    )
    if not body_names and mode in {"rigid", "static"}:
        body_names = ["asset_root"]
    for name in body_names:
        body = _balanced(links, links.find('"' + name + '"')) if name != "asset_root" else text
        objs = tuple(Path(item) for item in _OBJ.findall(body))
        visual = next((path for path in objs if path.name == "visual.obj"), None)
        collision = tuple(path for path in objs if path.name != "visual.obj")
        if visual is None or not collision:
            raise PackageError(f"{asset_id}:{name} lacks visual or collision OBJ")
        if mode == "static":
            bodies.append(
                OpenUsdRigidBody(
                    name,
                    visual,
                    collision,
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
                    collision,
                    _number(body, "physics:mass"),
                    _triple(body, "physics:centerOfMass"),
                    _triple(body, "physics:diagonalInertia"),
                    _quat(body, "physics:principalAxes"),
                )
            )
    joints = _read_joints(text) if mode == "articulated" else ()
    if mode == "articulated":
        _validate_tree(bodies, joints)
    materials = _read_shell_materials(path) if mode == "static" else ()
    return OpenUsdAsset(
        asset_id, digest, path.resolve(), mode, tuple(bodies), tuple(joints), materials
    )


def _read_shell_materials(path: Path) -> tuple[OpenUsdVisualMaterial, ...]:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise PackageError(f"cannot compose shell USD: {path}")
    materials: list[OpenUsdVisualMaterial] = []
    for name in ("Floor", "Walls"):
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
        materials.append(OpenUsdVisualMaterial(name.lower(), texture_path))
    return tuple(materials)


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


def _write_asset(path: Path, asset: OpenUsdAsset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": _asset_key(asset)})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "meshdir": "support/obj", "texturedir": "textures"},
    )
    assets = ET.SubElement(root, "asset")
    materials = {
        item.name: f"{item.name}_material" for item in asset.visual_materials
    }
    for item in asset.visual_materials:
        texture = f"{item.name}_texture"
        ET.SubElement(
            assets,
            "texture",
            {"name": texture, "type": "2d", "file": item.texture.with_suffix(".png").name},
        )
        ET.SubElement(assets, "material", {"name": materials[item.name], "texture": texture})
    world = ET.SubElement(root, "worldbody")
    bodies = {body.path: body for body in asset.bodies}
    children = {joint.child: joint for joint in asset.joints}
    roots = [body for body in asset.bodies if body.path not in children]
    if len(roots) != 1:
        raise PackageError(f"{asset.asset_id} does not have one articulation root")

    def emit(
        parent: ET.Element, body: OpenUsdRigidBody, joint: OpenUsdArticulationJoint | None
    ) -> None:
        attrs = {"name": "asset_root" if parent is world else body.path}
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
            else tuple((item, None) for item in _split_visual_obj(path, body.visual_obj))
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

    emit(world, roots[0], None)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _write_scene(
    path: Path,
    package: OpenUsdScenePackage,
    models: Mapping[tuple[str, str], str],
    mappings: dict[str, Any],
) -> None:
    root = ET.Element("mujoco", {"model": package.scene_id})
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
    ET.SubElement(
        world,
        "camera",
        {"name": "world_camera", "pos": "4 -4 3", "xyaxes": "0.707 0.707 0 -0.408 0.408 0.816"},
    )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _load_asset(path: Path) -> None:
    import mujoco

    mujoco.MjModel.from_xml_path(str(path))


def _validate_scene(path: Path, diagnostics: Path) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
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


def _write_preview(scene_path: Path, diagnostics: Path) -> str:
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        renderer.update_scene(data, camera="world_camera")
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
        target = textures / material.texture.with_suffix(".png").name
        with Image.open(material.texture) as image:
            image.convert("RGB").save(target)
        generated.append(target.relative_to(destination.parent.parent).as_posix())
    return generated


def _shell_visual_objs(
    asset_xml: Path,
    visual: Path,
    materials: tuple[OpenUsdVisualMaterial, ...],
) -> tuple[tuple[Path, str], ...]:
    if not materials:
        return ((visual, ""),)
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
