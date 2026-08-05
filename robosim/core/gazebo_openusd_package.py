"""Realize v9 OpenUSD resource packages as self-contained Gazebo Classic worlds."""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from robosim.core.csd import CsdRealizationManifest, make_csd_realization_cache_key
from robosim.core.mujoco_openusd_package import (
    OpenUsdArticulationJoint,
    OpenUsdAsset,
    OpenUsdRigidBody,
    OpenUsdScenePackage,
    PackageError,
    _dependency_hash,
    _unique_assets,
    read_openusd_scene_package,
)
from robosim.core.pybullet_openusd_package import _rotate, _rpy, _write_obj_materials


def compile_openusd_scene_package(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None,
    realization_version: str,
    simulator_version: str | None,
) -> CsdRealizationManifest:
    """Compile one validated v9 resource package to portable SDF 1.7."""
    package = read_openusd_scene_package(csd_path)
    cache = make_csd_realization_cache_key(
        csd_hash=_dependency_hash(package.root),
        asset_variant_hashes={
            asset.asset_id: asset.resource_digest for asset in _unique_assets(package)
        },
        backend="gazebo",
        realization_config=dict(realization_config or {}),
        realization_version=f"{realization_version}-gazebo-openusd-0.1",
        simulator_version=simulator_version,
    )
    root = (Path(output_root) / "gazebo" / package.scene_id).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = CsdRealizationManifest.from_json_dict(json.loads(manifest_path.read_text()))
        if manifest.cache_key == cache.digest and all(
            (root / item).is_file() for item in manifest.generated_files
        ):
            return manifest

    root.mkdir(parents=True, exist_ok=True)
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    generated = ["manifest.json", "world.sdf"]
    assets: dict[tuple[str, str], Path] = {}
    for asset in _unique_assets(package):
        key = _asset_key(asset)
        asset_root = root / "assets" / key
        shutil.copytree(asset.source.parent / "support", asset_root / "support", dirs_exist_ok=True)
        _write_obj_materials(asset_root, asset)
        assets[(asset.asset_id, asset.resource_digest)] = asset_root
        generated.extend(
            str(path.relative_to(root)) for path in asset_root.rglob("*") if path.is_file()
        )

    robot, robot_files = _robot_model(root, package)
    generated.extend(robot_files)
    _write_world(root / "world.sdf", package, assets, robot)
    generated.extend(
        str(path.relative_to(root))
        for path in (root / "assets").rglob("*")
        if path.is_file()
    )
    initial_state_file = _write_initial_state(root, package)
    if initial_state_file is not None:
        generated.append(initial_state_file)
    _validate(root / "world.sdf", diagnostics, package)
    generated.extend(("diagnostics/sdf_check.json", "diagnostics/entity_mapping.json"))
    manifest = CsdRealizationManifest(
        manifest_id=f"manifest_gazebo_{package.scene_id}",
        csd_id=package.scene_id,
        backend="gazebo",
        cache_key=cache.digest,
        root_path=str(root),
        entry_file="world.sdf",
        generated_files=tuple(dict.fromkeys(generated)),
        preview_files=(),
        initial_state_file=initial_state_file,
    )
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
    return manifest


def _asset_key(asset: OpenUsdAsset) -> str:
    return "asset-" + asset.resource_digest[:16]


def _write_world(
    path: Path,
    package: OpenUsdScenePackage,
    asset_roots: Mapping[tuple[str, str], Path],
    robot: ET.Element | None,
) -> None:
    root = ET.Element("sdf", {"version": "1.7"})
    world = ET.SubElement(root, "world", {"name": package.scene_id})
    physics = ET.SubElement(world, "physics", {"name": "default_physics", "type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"
    ET.SubElement(world, "gravity").text = "0 0 -9.81"
    for light in package.lights:
        element = ET.SubElement(world, "light", {"name": light.name, "type": "directional"})
        ET.SubElement(element, "diffuse").text = (
            f"{light.intensity} {light.intensity} {light.intensity} 1"
        )
        ET.SubElement(element, "direction").text = _values(
            _rotate(light.pose[3:], (0.0, 0.0, -1.0))
        )
    if robot is not None:
        world.append(robot)
    joint_states: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for instance in package.instances:
        key = (instance.asset.asset_id, instance.asset.resource_digest)
        model = _model(
            instance.asset,
            instance.prim_path.rsplit("/", 1)[-1],
            asset_roots[key],
            path.parent,
        )
        model.insert(0, _text("pose", _pose(instance.pose)))
        if instance.asset.mode == "static":
            model.insert(1, _text("static", "true"))
        world.append(model)
        if instance.joint_targets:
            joint_states.append((model.attrib["name"], instance.joint_targets))
    if joint_states:
        state = ET.SubElement(world, "state", {"world_name": package.scene_id})
        ET.SubElement(state, "iterations").text = "0"
        for model_name, targets in joint_states:
            model_state = ET.SubElement(state, "model", {"name": model_name})
            for joint_name, target in targets:
                joint = ET.SubElement(model_state, "joint", {"name": joint_name})
                ET.SubElement(joint, "angle", {"axis": "0"}).text = str(target)
    if package.cameras:
        sensors = ET.SubElement(world, "model", {"name": "csd_sensors"})
        ET.SubElement(sensors, "static").text = "true"
        link = ET.SubElement(sensors, "link", {"name": "sensors_link"})
        for camera in package.cameras:
            sensor = ET.SubElement(link, "sensor", {"name": camera.name, "type": "camera"})
            ET.SubElement(sensor, "pose").text = _pose(camera.pose)
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "30"
            config = ET.SubElement(sensor, "camera")
            ET.SubElement(config, "horizontal_fov").text = str(math.radians(camera.fovy))
            image = ET.SubElement(config, "image")
            ET.SubElement(image, "width").text = "512"
            ET.SubElement(image, "height").text = "512"
            ET.SubElement(image, "format").text = "R8G8B8"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _model(asset: OpenUsdAsset, name: str, asset_root: Path, world_root: Path) -> ET.Element:
    model = ET.Element("model", {"name": name})
    child_frames = {joint.child: _child_pose(joint) for joint in asset.joints}
    for body in asset.bodies:
        link = ET.SubElement(model, "link", {"name": body.path})
        if frame := child_frames.get(body.path):
            pose, parent = frame
            ET.SubElement(link, "pose", {"relative_to": parent}).text = _pose(pose)
        if asset.mode != "static":
            _inertial(link, body)
        for index, (visual, material) in enumerate(_visual_parts(asset_root, body, asset)):
            _geometry(link, "visual", visual, world_root, material, index)
        for index, collision in enumerate(body.collision_objs):
            element = _geometry(
                link,
                "collision",
                asset_root / collision,
                world_root,
                None,
                index,
            )
            surface = ET.SubElement(element, "surface")
            friction = ET.SubElement(ET.SubElement(surface, "friction"), "ode")
            ET.SubElement(friction, "mu").text = str(asset.dynamic_friction)
            ET.SubElement(friction, "mu2").text = str(asset.dynamic_friction)
    for joint in asset.joints:
        element = ET.SubElement(model, "joint", {"name": joint.name, "type": joint.kind})
        ET.SubElement(element, "parent").text = joint.parent
        ET.SubElement(element, "child").text = joint.child
        ET.SubElement(element, "pose", {"relative_to": joint.parent}).text = _pose(
            (*joint.parent_pos, *joint.parent_quat)
        )
        axis = ET.SubElement(element, "axis")
        ET.SubElement(axis, "xyz").text = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[joint.axis]
        limit = ET.SubElement(axis, "limit")
        ET.SubElement(limit, "lower").text = str(joint.limit[0])
        ET.SubElement(limit, "upper").text = str(joint.limit[1])
        dynamics = ET.SubElement(axis, "dynamics")
        ET.SubElement(dynamics, "damping").text = str(joint.damping)
    return model


def _inertial(link: ET.Element, body: OpenUsdRigidBody) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "pose").text = _pose((*body.center_of_mass, *body.principal_axes))
    ET.SubElement(inertial, "mass").text = str(body.mass)
    inertia = ET.SubElement(inertial, "inertia")
    for name, value in zip(("ixx", "iyy", "izz"), body.diagonal_inertia, strict=True):
        ET.SubElement(inertia, name).text = str(value)
    for name in ("ixy", "ixz", "iyz"):
        ET.SubElement(inertia, name).text = "0"


def _geometry(
    link: ET.Element,
    tag: str,
    mesh: Path,
    world_root: Path,
    material: tuple[float, float, float, float] | None,
    index: int,
) -> ET.Element:
    element = ET.SubElement(link, tag, {"name": f"{tag}_{index}"})
    geometry = ET.SubElement(element, "geometry")
    mesh_element = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh_element, "uri").text = mesh.relative_to(world_root).as_posix()
    if tag == "visual" and material is not None:
            sdf_material = ET.SubElement(element, "material")
            color = _values(material)
            ET.SubElement(sdf_material, "ambient").text = color
            ET.SubElement(sdf_material, "diffuse").text = color
    return element


def _visual_parts(
    asset_root: Path, body: OpenUsdRigidBody, asset: OpenUsdAsset
) -> tuple[tuple[Path, tuple[float, float, float, float] | None], ...]:
    source = asset_root / body.visual_obj
    material_by_name = {item.name: item.rgba for item in asset.visual_materials}
    names = _obj_material_names(source)
    if len(names) == 1:
        material = material_by_name.get(names[0] or "")
        if material is None:
            material = material_by_name.get(body.visual_obj.parent.name.lower())
        return ((source, material),)
    return tuple(
        (path, material_by_name.get(name or ""))
        for path, name in zip(_split_visual_obj(source), names, strict=True)
    )


def _obj_material_names(path: Path) -> tuple[str | None, ...]:
    names: list[str | None] = []
    material: str | None = None
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            if started:
                names.append(material)
            started = True
            material = None
        elif line.startswith("usemtl "):
            material = line.split(maxsplit=1)[1]
    return tuple((*names, material))


def _split_visual_obj(source: Path) -> tuple[Path, ...]:
    lines = source.read_text(encoding="utf-8").splitlines()
    groups: list[tuple[str | None, list[str]]] = []
    material: str | None = None
    faces: list[str] | None = None
    for line in lines:
        if line.startswith("o "):
            faces = []
            groups.append((None, faces))
            material = None
        elif line.startswith("usemtl "):
            material = line.split(maxsplit=1)[1]
            if groups:
                groups[-1] = (material, groups[-1][1])
        elif line.startswith("f ") and faces is not None:
            faces.append(line)
    if len(groups) <= 1:
        return (source,)
    prefix = [line for line in lines if not line.startswith(("o ", "usemtl ", "f "))]
    result: list[Path] = []
    for index, (name, group_faces) in enumerate(groups):
        if not group_faces:
            continue
        target = source.with_name(f"{source.stem}_{index:03d}.obj")
        material_line = f"usemtl {name}" if name else ""
        target.write_text("\n".join((*prefix, material_line, *group_faces, "")), encoding="utf-8")
        result.append(target)
    if len(result) != len(groups):
        raise PackageError(f"visual OBJ has empty object group: {source}")
    return tuple(result)


def _child_pose(
    joint: OpenUsdArticulationJoint,
) -> tuple[tuple[float, float, float, float, float, float, float], str]:
    parent = (*joint.parent_pos, *joint.parent_quat)
    child = (*joint.pos, *joint.quat)
    return _compose_pose(parent, _inverse_pose(child)), joint.parent


def _inverse_pose(
    pose: tuple[float, float, float, float, float, float, float]
) -> tuple[float, float, float, float, float, float, float]:
    position, quat = pose[:3], pose[3:]
    inverse_quat = (quat[0], -quat[1], -quat[2], -quat[3])
    inverse_position = _rotate(inverse_quat, (-position[0], -position[1], -position[2]))
    return (*inverse_position, *inverse_quat)


def _compose_pose(
    first: tuple[float, float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    position = _rotate(first[3:], second[:3])
    quaternion = _multiply(first[3:], second[3:])
    return (
        first[0] + position[0],
        first[1] + position[1],
        first[2] + position[2],
        *quaternion,
    )


def _multiply(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    fw, fx, fy, fz = first
    sw, sx, sy, sz = second
    return (
        fw * sw - fx * sx - fy * sy - fz * sz,
        fw * sx + fx * sw + fy * sz - fz * sy,
        fw * sy - fx * sz + fy * sw + fz * sx,
        fw * sz + fx * sy - fy * sx + fz * sw,
    )


def _write_initial_state(root: Path, package: OpenUsdScenePackage) -> str | None:
    targets = {
        f"{instance.prim_path.rsplit('/', 1)[-1]}/{joint}": value
        for instance in package.instances
        for joint, value in instance.joint_targets
    }
    if not targets:
        return None
    relative = "runtime/initial_joint_positions.json"
    path = root / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(targets, indent=2, sort_keys=True), encoding="utf-8")
    return relative


def _robot_model(
    root: Path, package: OpenUsdScenePackage
) -> tuple[ET.Element | None, tuple[str, ...]]:
    if package.robot is None:
        return None, ()
    if package.robot.robot_id != "franka_panda":
        raise PackageError(f"unsupported Gazebo robot: {package.robot.robot_id}")
    import pybullet_data

    source = Path(pybullet_data.getDataPath()) / "franka_panda"
    destination = root / "robots" / "franka_panda"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    urdf = ET.parse(destination / "panda.urdf").getroot()
    model = ET.Element("model", {"name": package.robot.instance_id})
    model.insert(0, _text("pose", _pose(package.robot.pose)))
    for urdf_link in urdf.findall("link"):
        link = ET.SubElement(model, "link", {"name": str(urdf_link.attrib["name"])})
        _copy_urdf_inertial(link, urdf_link)
        for tag in ("visual", "collision"):
            for index, element in enumerate(urdf_link.findall(tag)):
                _copy_urdf_geometry(link, tag, index, element, destination, root)
    for urdf_joint in urdf.findall("joint"):
        joint = ET.SubElement(
            model,
            "joint",
            {"name": str(urdf_joint.attrib["name"]), "type": str(urdf_joint.attrib["type"])},
        )
        parent = urdf_joint.find("parent")
        child = urdf_joint.find("child")
        if parent is None or child is None:
            raise PackageError(
                f"Gazebo Franka joint is missing parent or child: {joint.attrib['name']}"
            )
        ET.SubElement(joint, "parent").text = str(parent.attrib["link"])
        ET.SubElement(joint, "child").text = str(child.attrib["link"])
        origin = urdf_joint.find("origin")
        if origin is not None:
            ET.SubElement(joint, "pose").text = _urdf_pose(origin)
        axis = urdf_joint.find("axis")
        if axis is not None:
            sdf_axis = ET.SubElement(joint, "axis")
            ET.SubElement(sdf_axis, "xyz").text = str(axis.attrib["xyz"])
            limit = urdf_joint.find("limit")
            if limit is not None:
                sdf_limit = ET.SubElement(sdf_axis, "limit")
                ET.SubElement(sdf_limit, "lower").text = str(limit.attrib.get("lower", "0"))
                ET.SubElement(sdf_limit, "upper").text = str(limit.attrib.get("upper", "0"))
    return (
        model,
        tuple(str(path.relative_to(root)) for path in destination.rglob("*") if path.is_file()),
    )


def _validate(world: Path, diagnostics: Path, package: OpenUsdScenePackage) -> None:
    root = ET.parse(world).getroot()
    if root.attrib.get("version") != "1.7":
        raise PackageError("generated SDF does not declare version 1.7")
    missing = [
        str(uri.text)
        for uri in root.findall(".//mesh/uri")
        if not uri.text or not (world.parent / uri.text).is_file()
    ]
    payload = {"status": "passed", "checked_mesh_uris": len(root.findall(".//mesh/uri"))}
    (diagnostics / "sdf_check.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise PackageError(f"generated SDF has unresolved mesh URIs: {missing}")
    names = [item.prim_path.rsplit("/", 1)[-1] for item in package.instances]
    if package.robot is not None:
        names.append(package.robot.instance_id)
    (diagnostics / "entity_mapping.json").write_text(json.dumps({"models": names}, indent=2))


def _pose(values: tuple[float, float, float, float, float, float, float]) -> str:
    return _values(values[:3]) + " " + _rpy(values[3:])


def _values(values: tuple[float, ...]) -> str:
    return " ".join(str(value) for value in values)


def _text(tag: str, value: str) -> ET.Element:
    element = ET.Element(tag)
    element.text = value
    return element


def _copy_urdf_inertial(link: ET.Element, source: ET.Element) -> None:
    inertial = source.find("inertial")
    if inertial is None:
        return
    target = ET.SubElement(link, "inertial")
    origin = inertial.find("origin")
    if origin is not None:
        ET.SubElement(target, "pose").text = _urdf_pose(origin)
    mass = inertial.find("mass")
    if mass is not None:
        ET.SubElement(target, "mass").text = str(mass.attrib["value"])
    source_inertia = inertial.find("inertia")
    if source_inertia is not None:
        target_inertia = ET.SubElement(target, "inertia")
        for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            ET.SubElement(target_inertia, name).text = str(source_inertia.attrib[name])


def _copy_urdf_geometry(
    link: ET.Element, tag: str, index: int, source: ET.Element, robot_root: Path, world_root: Path
) -> None:
    target = ET.SubElement(link, tag, {"name": f"{tag}_{index}"})
    origin = source.find("origin")
    if origin is not None:
        ET.SubElement(target, "pose").text = _urdf_pose(origin)
    mesh = source.find("geometry/mesh")
    if mesh is None:
        return
    filename = str(mesh.attrib["filename"]).removeprefix("package://")
    local = robot_root / filename
    if not local.is_file():
        raise PackageError(f"Gazebo Franka mesh is unavailable: {filename}")
    geometry = ET.SubElement(target, "geometry")
    sdf_mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(sdf_mesh, "uri").text = local.relative_to(world_root).as_posix()
    if "scale" in mesh.attrib:
        ET.SubElement(sdf_mesh, "scale").text = str(mesh.attrib["scale"])


def _urdf_pose(origin: ET.Element) -> str:
    return f"{origin.attrib.get('xyz', '0 0 0')} {origin.attrib.get('rpy', '0 0 0')}"
