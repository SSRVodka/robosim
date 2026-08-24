"""Realize v9 OpenUSD resource packages for PyBullet."""

from __future__ import annotations

import hashlib
import json
import math
import os
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


def _require_package_output_root(package: OpenUsdScenePackage, output_root: Path) -> None:
    expected = package.root / "engine_manifests"
    if output_root.resolve() != expected.resolve():
        raise PackageError(
            f"output_root must be the scene package engine_manifests directory: {expected}"
        )


def _robot_hashes(package: OpenUsdScenePackage) -> dict[str, str]:
    if package.robot is None:
        return {}
    if package.robot.robot_id != "franka_panda":
        raise PackageError(f"unsupported PyBullet robot: {package.robot.robot_id}")
    import pybullet_data

    source = Path(pybullet_data.getDataPath()) / "franka_panda"
    if not (source / "panda.urdf").is_file():
        raise PackageError("PyBullet Franka template is unavailable")
    digest = hashlib.sha256()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return {f"robot:{package.robot.robot_id}": digest.hexdigest()}


def compile_openusd_scene_package(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None,
    realization_version: str,
    simulator_version: str | None,
) -> CsdRealizationManifest:
    """Compile one validated v9 resource package into a portable PyBullet package."""
    package = read_openusd_scene_package(csd_path)
    _require_package_output_root(package, output_root)
    robot_hashes = _robot_hashes(package)
    cache = make_csd_realization_cache_key(
        csd_hash=_dependency_hash(package.root),
        asset_variant_hashes=(
            {asset.asset_id: asset.resource_digest for asset in _unique_assets(package)}
            | robot_hashes
        ),
        backend="pybullet",
        realization_config=dict(realization_config or {}),
        realization_version=f"{realization_version}-pybullet-openusd-0.2",
        simulator_version=simulator_version,
    )
    root = (Path(output_root) / "pybullet" / package.scene_id).resolve()
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
    generated = ["manifest.json", "scene.py", "scene_meta.json"]
    asset_paths: dict[tuple[str, str], str] = {}
    for asset in _unique_assets(package):
        key = _asset_key(asset)
        relative = f"generated/assets/{key}.urdf"
        _write_urdf(root / relative, asset)
        asset_paths[(asset.asset_id, asset.resource_digest)] = relative
        generated.append(relative)
    meta = _meta(package, asset_paths)
    (root / "scene_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    _write_loader(root / "scene.py")
    _validate(root, package)
    generated.extend(("diagnostics/load.json", "diagnostics/physics.json"))
    preview = _write_preview(root, meta)
    if preview is not None:
        generated.append(preview)
    shutil.rmtree(root / "__pycache__", ignore_errors=True)
    manifest = CsdRealizationManifest(
        manifest_id=f"manifest_pybullet_{package.scene_id}",
        csd_id=package.scene_id,
        backend="pybullet",
        cache_key=cache.digest,
        root_path=str(root),
        entry_file="scene.py",
        generated_files=tuple(dict.fromkeys(generated)),
        preview_files=(preview,) if preview is not None else (),
    )
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
    return manifest


def _asset_key(asset: OpenUsdAsset) -> str:
    return "asset-" + asset.resource_digest[:16]


def _write_urdf(path: Path, asset: OpenUsdAsset) -> None:
    root = ET.Element("robot", {"name": _asset_key(asset)})
    joint_by_child = {joint.child: joint for joint in asset.joints}
    for body in asset.bodies:
        link = ET.SubElement(root, "link", {"name": body.path})
        frame = joint_by_child.get(body.path)
        _append_link(link, body, frame, asset.source.parent, path.parent)
    for joint in asset.joints:
        element = ET.SubElement(root, "joint", {"name": joint.name, "type": joint.kind})
        ET.SubElement(element, "parent", {"link": joint.parent})
        ET.SubElement(element, "child", {"link": joint.child})
        ET.SubElement(
            element, "origin", {"xyz": _values(joint.parent_pos), "rpy": _rpy(joint.parent_quat)}
        )
        ET.SubElement(element, "axis", {"xyz": _axis(joint.axis)})
        ET.SubElement(
            element,
            "limit",
            {
                "lower": str(joint.limit[0]),
                "upper": str(joint.limit[1]),
                "effort": "1000",
                "velocity": "10",
            },
        )
        if joint.damping:
            ET.SubElement(element, "dynamics", {"damping": str(joint.damping)})
    if asset.mode == "static" and len(asset.bodies) > 1:
        for body in asset.bodies[1:]:
            element = ET.SubElement(root, "joint", {"name": f"{body.path}_fixed", "type": "fixed"})
            ET.SubElement(element, "parent", {"link": asset.bodies[0].path})
            ET.SubElement(element, "child", {"link": body.path})
            ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _append_link(
    link: ET.Element,
    body: OpenUsdRigidBody,
    joint: OpenUsdArticulationJoint | None,
    asset_root: Path,
    urdf_root: Path,
) -> None:
    origin = {"xyz": "0 0 0", "rpy": "0 0 0"}
    if joint is not None:
        inverse = _inverse(joint.pos, joint.quat)
        origin = {"xyz": _values(inverse[0]), "rpy": _rpy(inverse[1])}
    inertial = ET.SubElement(link, "inertial")
    inertial_position, inertial_quat = _inertial_pose(body, joint)
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": _values(inertial_position), "rpy": _rpy(inertial_quat)},
    )
    ET.SubElement(inertial, "mass", {"value": str(body.mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(body.diagonal_inertia[0]),
            "ixy": "0",
            "ixz": "0",
            "iyy": str(body.diagonal_inertia[1]),
            "iyz": "0",
            "izz": str(body.diagonal_inertia[2]),
        },
    )
    _mesh(link, "visual", body.visual_obj, origin, asset_root, urdf_root)
    for index, collision in enumerate(body.collision_objs):
        _mesh(link, "collision", collision, origin, asset_root, urdf_root, index)


def _inertial_pose(
    body: OpenUsdRigidBody, joint: OpenUsdArticulationJoint | None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if joint is None:
        return body.center_of_mass, body.principal_axes
    frame_position, frame_quat = _inverse(joint.pos, joint.quat)
    return (
        _add(frame_position, _rotate(frame_quat, body.center_of_mass)),
        _multiply(frame_quat, body.principal_axes),
    )


def _mesh(
    link: ET.Element,
    tag: str,
    mesh: Path,
    origin: dict[str, str],
    asset_root: Path,
    urdf_root: Path,
    index: int = 0,
) -> None:
    element = ET.SubElement(link, tag, {"name": f"{tag}_{index}"})
    ET.SubElement(element, "origin", origin)
    geometry = ET.SubElement(element, "geometry")
    filename = Path(os.path.relpath(asset_root / mesh, urdf_root)).as_posix()
    ET.SubElement(geometry, "mesh", {"filename": filename})


def _meta(
    package: OpenUsdScenePackage, assets: Mapping[tuple[str, str], str]
) -> dict[str, object]:
    if len(package.lights) > 1:
        raise PackageError("PyBullet realization supports one DistantLight")
    instances = []
    for instance in package.instances:
        px, py, pz, qw, qx, qy, qz = instance.pose
        instances.append(
            {
                "name": instance.prim_path.rsplit("/", 1)[-1],
                "urdf_path": assets[(instance.asset.asset_id, instance.asset.resource_digest)],
                "static": instance.asset.mode == "static",
                "position": [px, py, pz],
                "orientation_xyzw": [qx, qy, qz, qw],
                "joint_targets": dict(instance.joint_targets),
                "dynamic_friction": instance.asset.dynamic_friction,
                "static_friction": instance.asset.static_friction,
                "restitution": instance.asset.restitution,
            }
        )
    cameras = []
    for camera in package.cameras:
        px, py, pz, qw, qx, qy, qz = camera.pose
        forward = _rotate((qw, qx, qy, qz), (0.0, 0.0, -1.0))
        up = _rotate((qw, qx, qy, qz), (0.0, 1.0, 0.0))
        cameras.append(
            {
                "name": camera.name,
                "position": [px, py, pz],
                "target": [px + forward[0], py + forward[1], pz + forward[2]],
                "up": list(up),
                "fovy": camera.fovy,
                "width": 512,
                "height": 512,
            }
        )
    result: dict[str, object] = {
        "backend": "pybullet",
        "csd_id": package.scene_id,
        "gravity": [0.0, 0.0, -9.81],
        "instances": instances,
        "cameras": cameras,
    }
    if package.lights:
        light = package.lights[0]
        direction = _rotate(light.pose[3:], (0.0, 0.0, -1.0))
        result["light"] = {"direction": list(direction), "intensity": light.intensity}
    if package.robot is not None:
        px, py, pz, qw, qx, qy, qz = package.robot.pose
        result.update(
            {
                "robot_name": package.robot.instance_id,
                "robot": {
                    "name": package.robot.instance_id,
                    "robot_id": package.robot.robot_id,
                    "position": [px, py, pz],
                    "orientation_xyzw": [qx, qy, qz, qw],
                },
            }
        )
    return result


def _write_loader(path: Path) -> None:
    loader = '''"""Generated PyBullet scene loader. Do not edit."""
from __future__ import annotations
import json
from pathlib import Path
import pybullet as p

def load_scene(physics_client_id: int) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    meta = json.loads((root / "scene_meta.json").read_text())
    p.setGravity(*meta["gravity"], physicsClientId=physics_client_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=physics_client_id)
    bodies = {}
    robot = meta.get("robot")
    if robot:
        import pybullet_data
        robot_path = Path(pybullet_data.getDataPath()) / robot["robot_id"] / "panda.urdf"
        bodies[meta["robot_name"]] = p.loadURDF(
            str(robot_path), basePosition=robot["position"],
            baseOrientation=robot["orientation_xyzw"], useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=physics_client_id,
        )
    for spec in meta["instances"]:
        body = p.loadURDF(
            str(root / spec["urdf_path"]), basePosition=spec["position"],
            baseOrientation=spec["orientation_xyzw"], useFixedBase=spec["static"],
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=physics_client_id,
        )
        bodies[spec["name"]] = body
        for link in range(-1, p.getNumJoints(body, physicsClientId=physics_client_id)):
            p.changeDynamics(
                body, link, lateralFriction=spec["dynamic_friction"],
                restitution=spec["restitution"], physicsClientId=physics_client_id,
            )
        for index in range(p.getNumJoints(body, physicsClientId=physics_client_id)):
            name = p.getJointInfo(body, index, physicsClientId=physics_client_id)[1].decode()
            target = spec["joint_targets"].get(name)
            if target is not None:
                p.resetJointState(
                    body, index, targetValue=target, physicsClientId=physics_client_id
                )
    return {"bodies": bodies, "metadata": meta}
'''
    path.write_text(loader, encoding="utf-8")


def _validate(root: Path, package: OpenUsdScenePackage) -> None:
    import importlib.util

    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        spec = importlib.util.spec_from_file_location("pybullet_scene", root / "scene.py")
        if spec is None or spec.loader is None:
            raise PackageError("cannot import generated PyBullet scene")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bodies = module.load_scene(client)["bodies"]
        if set(bodies) != {item.prim_path.rsplit("/", 1)[-1] for item in package.instances} | (
            {package.robot.instance_id} if package.robot else set()
        ):
            raise PackageError("PyBullet entity mapping differs from scene")
        for _ in range(20):
            p.stepSimulation(physicsClientId=client)
        (root / "diagnostics" / "load.json").write_text(
            json.dumps({"status": "passed", "bodies": len(bodies)})
        )
        (root / "diagnostics" / "physics.json").write_text(
            json.dumps({"status": "passed", "steps": 20})
        )
    finally:
        p.disconnect(client)


def _write_preview(root: Path, metadata: Mapping[str, object]) -> str | None:
    cameras = metadata.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return None
    camera = cameras[0]
    if not isinstance(camera, dict):
        raise PackageError("invalid PyBullet camera metadata")
    import importlib.util

    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        spec = importlib.util.spec_from_file_location("pybullet_preview", root / "scene.py")
        if spec is None or spec.loader is None:
            raise PackageError("cannot import generated PyBullet scene")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load_scene(client)
        width, height = int(camera["width"]), int(camera["height"])
        view = p.computeViewMatrix(camera["position"], camera["target"], camera["up"])
        projection = p.computeProjectionMatrixFOV(
            float(camera["fovy"]), float(width) / height, 0.01, 100.0
        )
        light = metadata.get("light", {})
        if not isinstance(light, dict):
            light = {}
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=light.get("direction", [1.0, 1.0, -1.0]),
            lightDiffuseCoeff=float(light.get("intensity", 1.0)),
            physicsClientId=client,
        )
        rgb = bytes(value for index, value in enumerate(bytes(rgba)) if index % 4 != 3)
        path = root / "diagnostics" / "preview.ppm"
        path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + rgb)
        return "diagnostics/preview.ppm"
    finally:
        p.disconnect(client)


def _axis(axis: str) -> str:
    return {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[axis]


def _values(values: tuple[float, float, float]) -> str:
    return " ".join(str(value) for value in values)


def _inverse(
    position: tuple[float, float, float], quat: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    inverse_quat = (quat[0], -quat[1], -quat[2], -quat[3])
    rotated = _rotate(inverse_quat, position)
    return ((-rotated[0], -rotated[1], -rotated[2]), inverse_quat)


def _rotate(
    quat: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    w, x, y, z = quat
    qv = (x, y, z)
    cross = _cross(qv, vector)
    t = (2 * cross[0], 2 * cross[1], 2 * cross[2])
    c = _cross(qv, t)
    return (
        vector[0] + w * t[0] + c[0],
        vector[1] + w * t[1] + c[1],
        vector[2] + w * t[2] + c[2],
    )


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _multiply(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rpy(quat: tuple[float, float, float, float]) -> str:
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return f"{roll} {pitch} {yaw}"
