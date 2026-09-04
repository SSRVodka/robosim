"""Realize v9 OpenUSD resource packages as self-contained Gazebo Classic worlds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robosim.core.csd import (
    CsdGazeboRuntimeContract,
    CsdRealizationManifest,
    make_csd_realization_cache_key,
)
from robosim.core.mujoco_openusd_package import (
    OpenUsdArticulationJoint,
    OpenUsdAsset,
    OpenUsdRigidBody,
    OpenUsdScenePackage,
    OpenUsdVisualMaterial,
    PackageError,
    _dependency_hash,
    _unique_assets,
    read_openusd_scene_package,
)
from robosim.core.pybullet_openusd_package import _rotate, _rpy

_Pose = tuple[float, float, float, float, float, float, float]
_RUNTIME_TEMPLATE_VERSION = "gazebo-runtime-3"
@dataclass(frozen=True, slots=True)
class _RobotAssetProfile:
    robot_id: str
    source: Path
    urdf_relative: Path
    srdf_relative: Path
    closure_roots: tuple[Path, ...]
    package_aliases: Mapping[str, Path]
    position_controllers: Mapping[str, str]
    cameras: tuple[Mapping[str, Any], ...]


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
    _require_package_output_root(package, output_root)
    if package.robot is None:
        raise PackageError("Gazebo runtime requires a robot")
    profile = _robot_asset_profile(package.robot.robot_id)
    try:
        runtime_hashes = _runtime_plugin_hashes(
            has_camera=bool(package.cameras or profile.cameras), robot_id=package.robot.robot_id
        )
    except FileNotFoundError as error:
        raise PackageError(str(error)) from error
    cache = make_csd_realization_cache_key(
        csd_hash=_dependency_hash(package.root),
        asset_variant_hashes={
            **{asset.asset_id: asset.resource_digest for asset in _unique_assets(package)},
            **runtime_hashes,
        },
        backend="gazebo",
        realization_config=dict(realization_config or {}),
        realization_version=f"{realization_version}-gazebo-openusd-0.9",
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
    generated = ["manifest.json", "world.sdf", "diagnostics/runtime_preflight.json"]
    assets: dict[tuple[str, str], Path] = {}
    for asset in _unique_assets(package):
        key = _asset_key(asset)
        asset_root = asset.source.parent
        material_root = root / "generated" / "materials" / key
        _write_gazebo_material_scripts(material_root, asset)
        assets[(asset.asset_id, asset.resource_digest)] = asset_root
        generated.extend(str(path.relative_to(root)) for path in material_root.glob("*.material"))

    robot, robot_files = _robot_model(root, package)
    generated.extend(robot_files)
    if robot is None:
        raise PackageError("Gazebo runtime requires a robot")
    try:
        runtime = _write_runtime_artifacts(
            root=root,
            csd_id=package.scene_id,
            urdf_source=profile.source / profile.urdf_relative,
            camera_names=(
                *(camera.name for camera in package.cameras),
                *(str(camera["name"]) for camera in profile.cameras),
            ),
            runtime_version=simulator_version,
        )
    except ValueError as error:
        raise PackageError(str(error)) from error
    _write_world(root / "world.sdf", package, assets, robot, runtime)
    generated.extend(
        (runtime.robot_control_urdf, runtime.controllers_file, runtime.robot_semantics_file)
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
        gazebo_runtime=runtime,
    )
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
    return manifest


def _asset_key(asset: OpenUsdAsset) -> str:
    return "asset-" + asset.resource_digest[:16]


def _runtime_plugin_hashes(*, has_camera: bool, robot_id: str = "franka_panda") -> dict[str, str]:
    """Return the exact installed runtime inputs required by this realization."""
    executables = ["gzserver", "ros2"]
    packages = [
        "robot_state_publisher",
        "gazebo_ros2_control",
        "controller_manager",
        "joint_state_broadcaster",
        "joint_trajectory_controller",
    ]
    libraries = [
        "libgazebo_ros2_control.so",
        "libjoint_state_broadcaster.so",
        "libjoint_trajectory_controller.so",
    ]
    if has_camera:
        libraries.append("libgazebo_ros_camera.so")
    hashes: dict[str, str] = {}
    missing = [name for name in executables if not (Path(sys.prefix) / "bin" / name).is_file()]
    missing.extend(
        package
        for package in packages
        if not (
            Path(sys.prefix) / "share" / "ament_index" / "resource_index" / "packages" / package
        ).is_file()
    )
    if missing:
        raise FileNotFoundError(f"required Gazebo ROS runtime is unavailable: {', '.join(missing)}")
    for library in libraries:
        path = Path(sys.prefix) / "lib" / library
        if not path.is_file():
            raise FileNotFoundError(f"required Gazebo ROS plugin is unavailable: {library}")
        hashes[f"runtime:{library}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["runtime:template"] = _RUNTIME_TEMPLATE_VERSION
    profile = _robot_asset_profile(robot_id)
    hashes[f"runtime:{robot_id}"] = _profile_closure_hash(profile)
    return hashes


def _require_package_output_root(package: OpenUsdScenePackage, output_root: Path) -> None:
    expected = package.root / "engine_manifests"
    if output_root.resolve() != expected.resolve():
        raise PackageError(
            f"output_root must be the scene package engine_manifests directory: {expected}"
        )


def _write_runtime_artifacts(
    *,
    root: Path,
    csd_id: str,
    urdf_source: Path,
    camera_names: Iterable[str],
    runtime_version: str | None = None,
) -> CsdGazeboRuntimeContract:
    namespace = f"/robosim/{csd_id}"
    control_urdf = root / "robot_control.urdf"
    urdf = ET.parse(urdf_source).getroot()
    joints = tuple(
        joint.attrib["name"]
        for joint in urdf.findall("joint")
        if joint.attrib["type"] != "fixed"
        and joint.find("limit") is not None
        and joint.find("mimic") is None
    )
    control = ET.SubElement(urdf, "ros2_control", {"name": "GazeboSystem", "type": "system"})
    hardware = ET.SubElement(control, "hardware")
    ET.SubElement(hardware, "plugin").text = "gazebo_ros2_control/GazeboSystem"
    for name in joints:
        joint = ET.SubElement(control, "joint", {"name": name})
        ET.SubElement(joint, "command_interface", {"name": "position"})
        ET.SubElement(joint, "state_interface", {"name": "position"})
        ET.SubElement(joint, "state_interface", {"name": "velocity"})
    ET.indent(urdf, space="  ")
    ET.ElementTree(urdf).write(control_urdf, encoding="utf-8", xml_declaration=True)

    controllers = root / "controllers.yaml"
    semantics_file, semantics = _write_robot_semantics(root, urdf_source)
    actions = {
        name: f"{namespace}/{controller}/follow_joint_trajectory"
        for name, controller in semantics["controller_names"].items()
    }
    controllers.write_text(
        _controllers_yaml(namespace, semantics["controller_groups"], semantics["controller_names"]),
        encoding="utf-8",
    )
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    (diagnostics / "runtime_preflight.json").write_text(
        json.dumps(
            {
                "runtime_prefix": sys.prefix,
                "runtime_version": runtime_version or sys.prefix,
                "gazebo_ros2_control": hashlib.sha256(
                    (Path(sys.prefix) / "lib" / "libgazebo_ros2_control.so").read_bytes()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    first_action = next(iter(actions.values()))
    return CsdGazeboRuntimeContract(
        world_file="world.sdf",
        robot_control_urdf=control_urdf.name,
        controllers_file=controllers.name,
        namespace=namespace,
        joint_state_topic=f"{namespace}/joint_states",
        trajectory_action=first_action,
        camera_topics={name: f"{namespace}/cameras/{name}/image_raw" for name in camera_names},
        gripper_trajectory_action=actions.get("hand", ""),
        robot_semantics_file=semantics_file,
        trajectory_actions=actions,
    )


def _append_control_plugin(model: ET.Element, runtime: CsdGazeboRuntimeContract) -> None:
    plugin = ET.SubElement(
        model,
        "plugin",
        {"name": "gazebo_ros2_control", "filename": "libgazebo_ros2_control.so"},
    )
    ET.SubElement(plugin, "robot_param").text = "robot_description"
    ET.SubElement(plugin, "robot_param_node").text = "robot_state_publisher"
    ET.SubElement(plugin, "parameters").text = runtime.controllers_file
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = runtime.namespace


def _append_camera_plugin(
    sensor: ET.Element, camera_name: str, runtime: CsdGazeboRuntimeContract
) -> None:
    topic = runtime.camera_topics.get(camera_name)
    if topic is None:
        return
    plugin = ET.SubElement(
        sensor,
        "plugin",
        {"name": f"{camera_name}_ros_camera", "filename": "libgazebo_ros_camera.so"},
    )
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = runtime.namespace
    ET.SubElement(plugin, "camera_name").text = f"cameras/{camera_name}"


def _camera_sensor(
    link: ET.Element,
    *,
    name: str,
    pose: str,
    fovy: float,
    width: int,
    height: int,
    runtime: CsdGazeboRuntimeContract,
) -> None:
    sensor = ET.SubElement(link, "sensor", {"name": name, "type": "camera"})
    ET.SubElement(sensor, "pose").text = pose
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = "30"
    config = ET.SubElement(sensor, "camera")
    ET.SubElement(config, "horizontal_fov").text = str(math.radians(fovy))
    image = ET.SubElement(config, "image")
    ET.SubElement(image, "width").text = str(width)
    ET.SubElement(image, "height").text = str(height)
    ET.SubElement(image, "format").text = "R8G8B8"
    _append_camera_plugin(sensor, name, runtime)


def _controllers_yaml(
    namespace: str, groups: Mapping[str, list[str]], names: Mapping[str, str]
) -> str:
    lines = [
        f"{namespace}/controller_manager:",
        "  ros__parameters:",
        "    update_rate: 100",
        "    joint_state_broadcaster:",
        "      type: joint_state_broadcaster/JointStateBroadcaster",
    ]
    for name in groups:
        lines.extend(
            (
                f"    {names[name]}:",
                "      type: joint_trajectory_controller/JointTrajectoryController",
            )
        )
    for name, joints in groups.items():
        lines.extend((f"{namespace}/{names[name]}:", "  ros__parameters:", "    joints:"))
        lines.extend(f"      - {joint}" for joint in joints)
        lines.extend(
            (
                "    command_interfaces:",
                "      - position",
                "    state_interfaces:",
                "      - position",
                "      - velocity",
            )
        )
    return "\n".join((*lines, ""))


def _write_world(
    path: Path,
    package: OpenUsdScenePackage,
    asset_roots: Mapping[tuple[str, str], Path],
    robot: ET.Element | None,
    runtime: CsdGazeboRuntimeContract,
) -> None:
    root = ET.Element("sdf", {"version": "1.7"})
    world = ET.SubElement(root, "world", {"name": package.scene_id})
    physics = ET.SubElement(world, "physics", {"name": "default_physics", "type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"
    ET.SubElement(world, "gravity").text = "0 0 -9.81"
    for light in package.lights:
        element = ET.SubElement(world, "light", {"name": light.name, "type": "directional"})
        ET.SubElement(
            element, "diffuse"
        ).text = f"{light.intensity} {light.intensity} {light.intensity} 1"
        ET.SubElement(element, "direction").text = _values(
            _rotate(light.pose[3:], (0.0, 0.0, -1.0))
        )
    if robot is not None:
        assert package.robot is not None
        _append_control_plugin(robot, runtime)
        profile = _robot_asset_profile(package.robot.robot_id)
        links = {link.attrib["name"]: link for link in robot.findall("link")}
        for robot_camera in profile.cameras:
            link_name = str(robot_camera["link"])
            if link_name not in links:
                raise PackageError(f"Gazebo robot camera link is unavailable: {link_name}")
            _camera_sensor(
                links[link_name],
                name=str(robot_camera["name"]),
                pose=" ".join(str(value) for value in robot_camera["pose"]),
                fovy=float(robot_camera["fovy"]),
                width=int(robot_camera["width"]),
                height=int(robot_camera["height"]),
                runtime=runtime,
            )
        world.append(robot)
    joint_states: list[tuple[str, _Pose, tuple[tuple[str, float], ...]]] = []
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
            joint_states.append((model.attrib["name"], instance.pose, instance.joint_targets))
    if joint_states:
        state = ET.SubElement(world, "state", {"world_name": package.scene_id})
        ET.SubElement(state, "iterations").text = "0"
        for model_name, pose, targets in joint_states:
            model_state = ET.SubElement(state, "model", {"name": model_name})
            ET.SubElement(model_state, "pose").text = _pose(pose)
            for joint_name, target in targets:
                joint = ET.SubElement(model_state, "joint", {"name": joint_name})
                ET.SubElement(joint, "angle", {"axis": "0"}).text = str(target)
    if package.cameras:
        sensors = ET.SubElement(world, "model", {"name": "csd_sensors"})
        ET.SubElement(sensors, "static").text = "true"
        link = ET.SubElement(sensors, "link", {"name": "sensors_link"})
        for scene_camera in package.cameras:
            _camera_sensor(
                link,
                name=scene_camera.name,
                pose=_pose(scene_camera.pose),
                fovy=scene_camera.fovy,
                width=512,
                height=512,
                runtime=runtime,
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _model(
    asset: OpenUsdAsset,
    name: str,
    asset_root: Path,
    world_root: Path,
) -> ET.Element:
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
            _geometry(link, "visual", visual, asset_root, world_root, asset, material, index)
        for index, collision in enumerate(body.collision_objs):
            element = _geometry(
                link,
                "collision",
                asset_root / collision,
                asset_root,
                world_root,
                asset,
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
        ET.SubElement(element, "pose", {"relative_to": joint.child}).text = _pose(
            (*joint.pos, *joint.quat)
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
    asset_root: Path,
    world_root: Path,
    asset: OpenUsdAsset,
    material: OpenUsdVisualMaterial | None,
    index: int,
) -> ET.Element:
    element = ET.SubElement(link, tag, {"name": f"{tag}_{index}"})
    geometry = ET.SubElement(element, "geometry")
    mesh_element = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh_element, "uri").text = mesh.relative_to(world_root).as_posix()
    if tag == "visual" and material is not None:
        sdf_material = ET.SubElement(element, "material")
        if material.texture is not None:
            script = ET.SubElement(sdf_material, "script")
            material_root = world_root / "generated" / "materials" / _asset_key(asset)
            ET.SubElement(script, "uri").text = material_root.relative_to(world_root).as_posix()
            ET.SubElement(script, "name").text = _gazebo_material_name(asset, material)
        else:
            color = _values(material.rgba)
            ET.SubElement(sdf_material, "ambient").text = color
            ET.SubElement(sdf_material, "diffuse").text = color
    return element


def _visual_parts(
    asset_root: Path, body: OpenUsdRigidBody, asset: OpenUsdAsset
) -> tuple[tuple[Path, OpenUsdVisualMaterial | None], ...]:
    source = asset_root / body.visual_obj
    material_by_name = {item.name: item for item in asset.visual_materials}
    names = _obj_material_names(source)
    if len(names) == 1:
        material = material_by_name.get(names[0] or "")
        if material is None:
            material = material_by_name.get(body.visual_obj.parent.name.lower())
        return ((source, material),)
    fallback = material_by_name.get(body.visual_obj.parent.name.lower())
    return tuple(
        (path, material_by_name.get(name or "", fallback))
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
    pose: tuple[float, float, float, float, float, float, float],
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
    profile = _robot_asset_profile(package.robot.robot_id)
    destination = profile.source
    urdf = ET.parse(destination / profile.urdf_relative).getroot()
    model = ET.Element("model", {"name": package.robot.instance_id})
    model.insert(0, _text("pose", _pose(package.robot.pose)))
    massless_links = _massless_fixed_urdf_links(urdf)
    joints = _retained_urdf_joints(urdf, massless_links)
    child_frames: dict[str, tuple[str, tuple[float, float, float, float, float, float, float]]] = {}
    for joint, parent, pose in joints:
        child = joint.find("child")
        if child is None:
            raise PackageError(f"Gazebo Franka joint is missing child: {joint.attrib['name']}")
        child_frames[str(child.attrib["link"])] = (parent, pose)
    for urdf_link in urdf.findall("link"):
        if urdf_link.attrib["name"] in massless_links:
            continue
        link = ET.SubElement(model, "link", {"name": str(urdf_link.attrib["name"])})
        if frame := child_frames.get(str(urdf_link.attrib["name"])):
            pose, parent = frame[1], frame[0]
            ET.SubElement(link, "pose", {"relative_to": parent}).text = _urdf_pose_text(pose)
        _copy_urdf_inertial(link, urdf_link)
        for tag in ("visual", "collision"):
            for index, element in enumerate(urdf_link.findall(tag)):
                _copy_urdf_geometry(link, tag, index, element, destination, root)
    if package.robot.fixed_base:
        children = {
            child.attrib["link"]
            for joint in urdf.findall("joint")
            if (child := joint.find("child")) is not None
        }
        roots = [
            link.attrib["name"]
            for link in urdf.findall("link")
            if link.attrib["name"] not in children
        ]
        if len(roots) != 1:
            raise PackageError(f"Gazebo robot has no unique base link: {package.robot.robot_id}")
        base_joint = ET.SubElement(model, "joint", {"name": "robot_base_fixed", "type": "fixed"})
        ET.SubElement(base_joint, "parent").text = "world"
        ET.SubElement(base_joint, "child").text = roots[0]
    for urdf_joint, parent_name, _ in joints:
        joint = ET.SubElement(
            model,
            "joint",
            {"name": str(urdf_joint.attrib["name"]), "type": str(urdf_joint.attrib["type"])},
        )
        child = urdf_joint.find("child")
        if child is None:
            raise PackageError(
                f"Gazebo Franka joint is missing parent or child: {joint.attrib['name']}"
            )
        ET.SubElement(joint, "parent").text = parent_name
        ET.SubElement(joint, "child").text = str(child.attrib["link"])
        ET.SubElement(
            joint, "pose", {"relative_to": str(child.attrib["link"])}
        ).text = "0 0 0 0 0 0"
        axis = urdf_joint.find("axis")
        if axis is not None:
            sdf_axis = ET.SubElement(joint, "axis")
            ET.SubElement(sdf_axis, "xyz").text = str(axis.attrib["xyz"])
            limit = urdf_joint.find("limit")
            if limit is not None:
                sdf_limit = ET.SubElement(sdf_axis, "limit")
                ET.SubElement(sdf_limit, "lower").text = str(limit.attrib.get("lower", "0"))
                ET.SubElement(sdf_limit, "upper").text = str(limit.attrib.get("upper", "0"))
    return model, ()


def _robot_asset_profile(robot_id: str) -> _RobotAssetProfile:
    robot_root = (
        Path(__file__).resolve().parents[2]
        / "drivers_sim"
        / "gazebo-11"
        / "assets"
        / "robots"
    )
    try:
        registry = json.loads((robot_root / "manifest.json").read_text(encoding="utf-8"))
        entries = registry["robots"]
        directory = next(
            entry["directory"]
            for entry in entries
            if entry["robot_id"] == robot_id
        )
    except (KeyError, OSError, StopIteration, TypeError, json.JSONDecodeError) as error:
        raise PackageError(f"Gazebo robot asset profile is unavailable: {robot_id}") from error
    if (
        registry.get("schema_version") != "robosim.robot-assets/1"
        or not isinstance(directory, str)
        or Path(directory).is_absolute()
        or ".." in Path(directory).parts
    ):
        raise PackageError(f"Gazebo robot asset profile is unavailable: {robot_id}")
    source = robot_root / directory
    try:
        payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"Gazebo robot asset profile is unavailable: {robot_id}") from error
    gazebo = payload.get("gazebo")
    if (
        payload.get("schema_version") != "robosim.robot-asset/1"
        or payload.get("robot_id") != robot_id
        or not isinstance(gazebo, dict)
    ):
        raise PackageError(f"invalid Gazebo robot asset profile: {robot_id}")
    urdf = Path(str(gazebo.get("urdf", "")))
    srdf = Path(str(gazebo.get("srdf", "")))
    roots = tuple(Path(str(item)) for item in gazebo.get("closure_roots", ()))
    aliases = {
        str(key): Path(str(value)) for key, value in dict(gazebo.get("package_aliases", {})).items()
    }
    entries = gazebo.get("position_controllers", ())
    cameras = gazebo.get("cameras", [])
    if not isinstance(entries, list):
        raise PackageError(f"invalid Gazebo robot controllers: {robot_id}")
    controllers = {
        str(item["group"]): str(item["controller"])
        for item in entries
        if isinstance(item, dict)
        and isinstance(item.get("group"), str)
        and isinstance(item.get("controller"), str)
    }
    relatives = (urdf, srdf, *roots, *aliases.values())
    if (
        not urdf.parts
        or not srdf.parts
        or not roots
        or len(controllers) != len(entries)
        or len(set(controllers.values())) != len(controllers)
        or any(path.is_absolute() or ".." in path.parts for path in relatives)
        or not isinstance(cameras, list)
        or any(
            not isinstance(camera, dict)
            or not isinstance(camera.get("name"), str)
            or not isinstance(camera.get("link"), str)
            or not isinstance(camera.get("pose"), list)
            or len(camera["pose"]) != 6
            or not all(key in camera for key in ("fovy", "width", "height"))
            for camera in cameras
        )
    ):
        raise PackageError(f"incomplete Gazebo robot asset profile: {robot_id}")
    if not (source / urdf).is_file() or not (source / srdf).is_file():
        raise PackageError(f"Gazebo robot profile has unresolved semantic files: {robot_id}")
    if any(not (source / root).is_dir() for root in roots):
        raise PackageError(f"Gazebo robot profile has unresolved closure roots: {robot_id}")
    if any(not (source / target).is_dir() for target in aliases.values()):
        raise PackageError(f"Gazebo robot profile has unresolved package alias: {robot_id}")
    return _RobotAssetProfile(
        robot_id=robot_id,
        source=source,
        urdf_relative=urdf,
        srdf_relative=srdf,
        closure_roots=roots,
        package_aliases=aliases,
        position_controllers=controllers,
        cameras=tuple(cameras),
    )


def _profile_closure_hash(profile: _RobotAssetProfile) -> str:
    """Hash only files the manifest declares part of this robot realization."""
    digest = hashlib.sha256()
    files = [profile.source / "manifest.json"]
    files.extend(
        path
        for root in profile.closure_roots
        for path in (profile.source / root).rglob("*")
        if path.is_file()
    )
    for path in sorted(files):
        digest.update(path.relative_to(profile.source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _franka_gazebo_source() -> Path:
    return _robot_asset_profile("franka_panda").source


def _write_robot_semantics(root: Path, urdf_source: Path) -> tuple[str, dict[str, Any]]:
    robot_root = next(
        (parent for parent in urdf_source.parents if (parent / "manifest.json").is_file()),
        None,
    )
    if robot_root is None:
        raise ValueError("Gazebo robot manifest is unavailable")
    profile = json.loads((robot_root / "manifest.json").read_text(encoding="utf-8"))["gazebo"]
    srdf = robot_root / str(profile["srdf"])
    urdf = ET.parse(urdf_source).getroot()
    limits = {
        joint.attrib["name"]: {
            "type": joint.attrib["type"],
            "lower": float(limit.attrib.get("lower", 0.0)),
            "upper": float(limit.attrib.get("upper", 0.0)),
            "velocity": float(limit.attrib.get("velocity", 0.0)),
            "effort": float(limit.attrib.get("effort", 0.0)),
        }
        for joint in urdf.findall("joint")
        if joint.attrib["type"] != "fixed" and (limit := joint.find("limit")) is not None
    }
    mimic_joints = {
        joint.attrib["name"] for joint in urdf.findall("joint") if joint.find("mimic") is not None
    }
    groups = _srdf_groups(srdf, urdf, set(limits))
    controller_names = {
        str(item["group"]): str(item["controller"])
        for item in profile["position_controllers"]
        if isinstance(item, dict)
    }
    controller_groups = {name: groups[name] for name in controller_names if name in groups}
    if len(controller_groups) != len(controller_names):
        raise PackageError(f"Gazebo controller group is absent from SRDF: {urdf.attrib['name']}")
    if any(
        not joints or mimic_joints.intersection(joints) for joints in controller_groups.values()
    ):
        raise PackageError(
            f"Gazebo controller group contains a mimic or no movable joint: {urdf.attrib['name']}"
        )
    semantic = {
        "robot_name": urdf.attrib["name"],
        "srdf_file": Path(os.path.relpath(srdf, root)).as_posix(),
        "joint_limits": limits,
        "mimic_joints": sorted(mimic_joints),
        "groups": groups,
        "controller_groups": controller_groups,
        "controller_names": controller_names,
        "named_states": _srdf_named_states(srdf, groups),
        "end_effectors": [
            dict(item.attrib) for item in ET.parse(srdf).getroot().findall("end_effector")
        ],
    }
    relative = "runtime/robot_semantics.json"
    path = root / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(semantic, indent=2, sort_keys=True), encoding="utf-8")
    return relative, semantic


def _srdf_named_states(
    path: Path, groups: Mapping[str, list[str]]
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for state in ET.parse(path).getroot().findall("group_state"):
        group = state.attrib["group"]
        values = {
            joint.attrib["name"]: float(joint.attrib["value"]) for joint in state.findall("joint")
        }
        joint_names = groups.get(group, [])
        result.setdefault(group, {})[state.attrib["name"]] = [
            values.get(name, 0.0) for name in joint_names
        ]
    return result


def _srdf_groups(path: Path, urdf: ET.Element, movable: set[str]) -> dict[str, list[str]]:
    parent_joint = {
        child.attrib["link"]: joint
        for joint in urdf.findall("joint")
        if (child := joint.find("child")) is not None
    }
    raw = {group.attrib["name"]: group for group in ET.parse(path).getroot().findall("group")}

    def resolve(name: str) -> list[str]:
        if name not in raw:
            raise PackageError(f"SRDF group references unknown subgroup '{name}'")
        group = raw[name]
        result = [
            item.attrib["name"] for item in group.findall("joint") if item.attrib["name"] in movable
        ]
        for chain in group.findall("chain"):
            link = chain.attrib["tip_link"]
            chain_joints: list[str] = []
            while link != chain.attrib["base_link"]:
                joint = parent_joint.get(link)
                parent = joint.find("parent") if joint is not None else None
                if joint is None or parent is None:
                    raise PackageError(f"SRDF chain cannot resolve group '{name}'")
                if joint.attrib["name"] in movable:
                    chain_joints.append(joint.attrib["name"])
                link = parent.attrib["link"]
            result.extend(reversed(chain_joints))
        for child in group.findall("group"):
            result.extend(resolve(child.attrib["name"]))
        return list(dict.fromkeys(result))

    groups = {name: resolve(name) for name in raw}
    return {name: joints for name, joints in groups.items() if joints}


def _massless_fixed_urdf_links(urdf: ET.Element) -> frozenset[str]:
    """Return zero-mass fixed connector links that Gazebo ODE cannot simulate."""
    joints_by_child = {
        str(child.attrib["link"]): joint
        for joint in urdf.findall("joint")
        if (child := joint.find("child")) is not None
    }
    result: set[str] = set()
    for link in urdf.findall("link"):
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        name = str(link.attrib["name"])
        joint = joints_by_child.get(name)
        if mass is None:
            if (
                joint is not None
                and joint.attrib["type"] == "fixed"
                and link.find("visual") is None
                and link.find("collision") is None
            ):
                result.add(name)
            continue
        if float(mass.attrib["value"]) != 0.0:
            continue
        if joint is None:
            continue
        if (
            joint.attrib["type"] != "fixed"
            or link.find("visual") is not None
            or link.find("collision") is not None
        ):
            raise PackageError(f"Gazebo robot has an unsupported zero-mass link: {name}")
        result.add(name)
    return frozenset(result)


def _retained_urdf_joints(
    urdf: ET.Element, massless_links: frozenset[str]
) -> tuple[tuple[ET.Element, str, tuple[float, float, float, float, float, float, float]], ...]:
    """Collapse removed fixed connector links into their retained descendants."""
    joints_by_child = {
        str(child.attrib["link"]): joint
        for joint in urdf.findall("joint")
        if (child := joint.find("child")) is not None
    }
    result = []
    for joint in urdf.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            raise PackageError(
                f"Gazebo Franka joint is missing parent or child: {joint.attrib['name']}"
            )
        if child.attrib["link"] in massless_links:
            continue
        parent_name = str(parent.attrib["link"])
        pose = _urdf_joint_pose(joint.find("origin"))
        while parent_name in massless_links:
            connector = joints_by_child[parent_name]
            connector_parent = connector.find("parent")
            if connector_parent is None:
                raise PackageError(f"Gazebo Franka connector has no parent: {parent_name}")
            pose = _compose_urdf_poses(_urdf_joint_pose(connector.find("origin")), pose)
            parent_name = str(connector_parent.attrib["link"])
        result.append((joint, parent_name, pose))
    return tuple(result)


def _urdf_joint_pose(
    origin: ET.Element | None,
) -> tuple[float, float, float, float, float, float, float]:
    if origin is None:
        return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    x, y, z = (float(value) for value in origin.attrib.get("xyz", "0 0 0").split())
    roll, pitch, yaw = (float(value) for value in origin.attrib.get("rpy", "0 0 0").split())
    half_roll, half_pitch, half_yaw = roll / 2, pitch / 2, yaw / 2
    quaternion = (
        math.cos(half_roll) * math.cos(half_pitch) * math.cos(half_yaw)
        + math.sin(half_roll) * math.sin(half_pitch) * math.sin(half_yaw),
        math.sin(half_roll) * math.cos(half_pitch) * math.cos(half_yaw)
        - math.cos(half_roll) * math.sin(half_pitch) * math.sin(half_yaw),
        math.cos(half_roll) * math.sin(half_pitch) * math.cos(half_yaw)
        + math.sin(half_roll) * math.cos(half_pitch) * math.sin(half_yaw),
        math.cos(half_roll) * math.cos(half_pitch) * math.sin(half_yaw)
        - math.sin(half_roll) * math.sin(half_pitch) * math.cos(half_yaw),
    )
    return (x, y, z, *quaternion)


def _compose_urdf_poses(
    parent: tuple[float, float, float, float, float, float, float],
    child: tuple[float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    position = _rotate(parent[3:], child[:3])
    return (
        parent[0] + position[0],
        parent[1] + position[1],
        parent[2] + position[2],
        *_multiply(parent[3:], child[3:]),
    )


def _urdf_pose_text(pose: tuple[float, float, float, float, float, float, float]) -> str:
    return _values(pose[:3]) + " " + _rpy(pose[3:])


def _write_collision_obj_materials(asset_root: Path, asset: OpenUsdAsset) -> None:
    """Bind collision OBJ shape names to local default materials for Gazebo Classic."""
    visual_paths = {body.visual_obj for body in asset.bodies}
    for body in asset.bodies:
        for collision in body.collision_objs:
            if collision in visual_paths:
                continue
            path = asset_root / collision
            lines = path.read_text(encoding="utf-8").splitlines()
            names = tuple(
                dict.fromkeys(line.split(maxsplit=1)[1] for line in lines if line.startswith("o "))
            )
            if not names:
                continue
            mtl = path.with_suffix(".mtl")
            mtl.write_text(
                "\n".join(f"newmtl {name}\nKd 0.7 0.7 0.7\nd 1\n" for name in names),
                encoding="utf-8",
            )
            obj_lines: list[str] = [f"mtllib {mtl.name}"]
            for line in lines:
                if line.startswith(("mtllib ", "usemtl ")):
                    continue
                obj_lines.append(line)
                if line.startswith("o "):
                    obj_lines.append(f"usemtl {line.split(maxsplit=1)[1]}")
            path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")


def _write_gazebo_material_scripts(material_root: Path, asset: OpenUsdAsset) -> None:
    """Write OGRE scripts that make textured USD materials authoritative in Gazebo."""
    for material in asset.visual_materials:
        if material.texture is None:
            continue
        texture = material.texture
        if not texture.is_file():
            raise PackageError(f"Gazebo texture is unavailable: {material.texture}")
        script = material_root / f"{_material_file_name(material.name)}.material"
        script.parent.mkdir(exist_ok=True)
        color = " ".join(str(value) for value in material.rgba)
        script.write_text(
            "\n".join(
                (
                    f"material {_gazebo_material_name(asset, material)}",
                    "{",
                    "  technique",
                    "  {",
                    "    pass",
                    "    {",
                    f"      ambient {color}",
                    f"      diffuse {color}",
                    "      texture_unit",
                    "      {",
                    f"        texture {Path(os.path.relpath(texture, material_root)).as_posix()}",
                    "      }",
                    "    }",
                    "  }",
                    "}",
                    "",
                )
            ),
            encoding="utf-8",
        )


def _gazebo_material_name(asset: OpenUsdAsset, material: OpenUsdVisualMaterial) -> str:
    return f"robosim/{_asset_key(asset)}/{_material_file_name(material.name)}"


def _material_file_name(name: str) -> str:
    return name.replace("/", "_")


def _normalize_robot_obj_materials(robot_root: Path) -> None:
    """Make copied Franka OBJ material references package-local and resolvable."""
    textures = tuple(robot_root.rglob("meshes/visual/colors.png"))
    if not textures:
        return
    if len(textures) != 1:
        raise PackageError("Gazebo robot OBJ closure has no unique color texture")
    texture = textures[0]
    for obj in robot_root.rglob("*.obj"):
        lines = obj.read_text(encoding="utf-8").splitlines()
        names = tuple(
            dict.fromkeys(line.split(maxsplit=1)[1] for line in lines if line.startswith("usemtl "))
        )
        if not names:
            continue
        mtl = obj.with_suffix(".mtl")
        if mtl.is_file():
            records = _normalize_mtl_texture_paths(mtl.read_text(encoding="utf-8"), mtl, texture)
        else:
            records = "\n".join(f"newmtl {name}\nKd 0.7 0.7 0.7\nd 1\n" for name in names)
        mtl.write_text(records, encoding="utf-8")
        body = tuple(line for line in lines if not line.startswith("mtllib "))
        obj.write_text(
            "\n".join((f"mtllib {mtl.name}", *body)) + "\n",
            encoding="utf-8",
        )


def _normalize_mtl_texture_paths(source: str, mtl: Path, texture: Path) -> str:
    """Replace non-local texture paths in a copied MTL with the copied texture."""
    texture_ref = Path(os.path.relpath(texture, mtl.parent)).as_posix()
    lines = source.splitlines()
    return (
        "\n".join(f"map_Kd {texture_ref}" if line.startswith("map_Kd ") else line for line in lines)
        + "\n"
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
    if not local.is_file() and "/" in filename:
        alias, relative = filename.split("/", 1)
        profile = json.loads((robot_root / "manifest.json").read_text(encoding="utf-8"))
        aliases = dict(profile["gazebo"]["package_aliases"])
        if alias in aliases:
            local = robot_root / str(aliases[alias]) / relative
    if not local.is_file():
        raise PackageError(f"Gazebo Franka mesh is unavailable: {filename}")
    geometry = ET.SubElement(target, "geometry")
    sdf_mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(sdf_mesh, "uri").text = local.relative_to(world_root).as_posix()
    if "scale" in mesh.attrib:
        ET.SubElement(sdf_mesh, "scale").text = str(mesh.attrib["scale"])


def _urdf_pose(origin: ET.Element) -> str:
    return f"{origin.attrib.get('xyz', '0 0 0')} {origin.attrib.get('rpy', '0 0 0')}"
