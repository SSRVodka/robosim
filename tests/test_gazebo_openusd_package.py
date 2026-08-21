"""Tests for v9 OpenUSD resource-package realization in Gazebo."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from robosim.core.csd_compiler import compile_csd
from robosim.core.gazebo_openusd_package import (
    _robot_asset_profile,
    _robot_model,
    _write_runtime_artifacts,
)
from robosim.core.mujoco_openusd_package import (
    PackageError,
    _read_robot,
    read_openusd_scene_package,
)

BENCHMARK_SCENE = Path(__file__).parents[1] / "csd" / "benchmark_gen" / "scene.usda"


def test_compile_v9_openusd_package_to_self_contained_gazebo_world(tmp_path: Path) -> None:
    result = compile_csd(
        backend="gazebo",
        csd_path=BENCHMARK_SCENE,
        output_root=tmp_path / "engine_manifests",
        simulator_version="test-gazebo",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    root = Path(result.manifest.root_path)
    assert result.manifest.gazebo_runtime is not None
    runtime = result.manifest.gazebo_runtime
    assert runtime.namespace == "/robosim/csd_4c9f31903d8cf0dc"
    assert runtime.camera_topics == {
        "agent_view": f"{runtime.namespace}/cameras/agent_view/image_raw",
        "wrist_view": f"{runtime.namespace}/cameras/wrist_view/image_raw",
    }
    assert (root / runtime.robot_control_urdf).is_file()
    assert (root / runtime.controllers_file).is_file()
    assert (root / runtime.robot_semantics_file).is_file()
    semantics = json.loads(
        (root / runtime.robot_semantics_file).read_text(encoding="utf-8")
    )
    assert (
        root / "robots" / "franka_panda" / "panda_moveit_config" / "config" / "panda.srdf"
    ).is_file()
    robot_urdf = ET.parse(root / runtime.robot_control_urdf).getroot()
    assert all(link.find("inertial") is not None for link in robot_urdf.findall("link"))
    assert all(
        str(mesh.attrib["filename"]).endswith(".obj")
        for mesh in robot_urdf.findall("link/visual/geometry/mesh")
    )
    assert runtime.gripper_trajectory_action.endswith("hand_controller/follow_joint_trajectory")
    assert "panda_finger_joint2" in semantics["mimic_joints"]
    control_urdf = robot_urdf
    assert control_urdf.find("ros2_control/joint[@name='panda_finger_joint2']") is None
    assert semantics["groups"] == {
        "panda_arm": [f"panda_joint{index}" for index in range(1, 8)],
        "hand": ["panda_finger_joint1"],
        "panda_arm_hand": [
            *(f"panda_joint{index}" for index in range(1, 8)),
            "panda_finger_joint1",
        ],
    }
    assert f"{runtime.namespace}/controller_manager:" in (
        root / runtime.controllers_file
    ).read_text(encoding="utf-8")
    assert "hand_controller:" in (root / runtime.controllers_file).read_text(encoding="utf-8")
    world = ET.parse(root / "world.sdf").getroot().find("world")
    assert world is not None
    assert ET.parse(root / "world.sdf").getroot().attrib["version"] == "1.7"
    models = {model.attrib["name"]: model for model in world.findall("model")}
    assert {
        "robot",
        "room_4x4_empty_square",
        "cabinet_double_door_01",
        "stool_square_low_01",
    } <= set(models)
    wrist = models["robot"].find("link[@name='panda_hand']/sensor[@name='wrist_view']")
    assert wrist is not None
    assert wrist.findtext("plugin/camera_name") == "cameras/wrist_view"
    agent = world.find("model[@name='csd_sensors']/link/sensor[@name='agent_view']")
    assert agent is not None
    assert agent.findtext("plugin/camera_name") == "cameras/agent_view"
    cabinet = models["cabinet_double_door_01"]
    assert cabinet.findall("joint")
    assert len(cabinet.findall(".//collision")) > 1
    assert len(cabinet.findall(".//visual")) > len(cabinet.findall("link"))
    assert len({item.text for item in cabinet.findall(".//visual/material/diffuse")}) > 1
    assert world.find("state/model[@name='cabinet_double_door_01']/joint/angle") is not None
    shell = models["room_4x4_empty_square"]
    assert {link.attrib["name"] for link in shell.findall("link")} == {"Floor", "Walls"}
    assert all(link.find("inertial") is None for link in shell.findall("link"))
    floor = shell.find("link[@name='Floor']")
    walls = shell.find("link[@name='Walls']")
    assert floor is not None and walls is not None
    floor_material = floor.findtext("visual/material/script/name")
    assert floor_material is not None and floor_material.endswith("/floor")
    assert len(walls.findall("visual")) == 4
    assert all(
        (material := visual.findtext("material/script/name")) is not None
        and material.endswith("/walls")
        for visual in walls.findall("visual")
    )
    for joint in cabinet.findall("joint"):
        child = joint.findtext("child")
        pose = joint.find("pose")
        assert child is not None and pose is not None
        assert pose.attrib["relative_to"] == child
    assert cabinet.find("joint[@name='asset_root_fixed']") is None
    state_pose = world.findtext("state/model[@name='cabinet_double_door_01']/pose")
    assert state_pose == cabinet.findtext("pose")
    robot = models["robot"]
    robot_link_names = {link.attrib["name"] for link in robot.findall("link")}
    assert "panda_link8" not in robot_link_names
    assert "panda_grasptarget" not in robot_link_names
    assert (
        robot.find("joint[@name='robot_base_fixed'][parent='world'][child='panda_link0']")
        is not None
    )
    assert robot.find("joint[parent='panda_link7'][child='panda_hand']") is not None
    assert robot.find("plugin[@filename='libgazebo_ros2_control.so']") is not None
    assert robot.find("link[@name='panda_link1']/pose[@relative_to='panda_link0']") is not None
    assert all(float(str(mass.text)) > 0.0 for mass in robot.findall("link/inertial/mass"))
    textured_visuals = [
        visual
        for visual in world.findall(".//visual")
        if visual.find("material/script") is not None
    ]
    assert textured_visuals
    for visual in textured_visuals:
        script = visual.find("material/script")
        assert script is not None
        script_root = root / str(script.findtext("uri"))
        assert script_root.is_dir()
        assert any(script_root.glob("*.material"))
        assert visual.find("material/diffuse") is None
    for obj in (root / "robots" / "franka_panda" / "panda_description").rglob("*.obj"):
        references = [
            line.split(maxsplit=1)[1]
            for line in obj.read_text(encoding="utf-8").splitlines()
            if line.startswith("mtllib ")
        ]
        assert references == [f"{obj.stem}.mtl"]
        mtl = obj.with_suffix(".mtl")
        assert mtl.is_file()
        for line in mtl.read_text(encoding="utf-8").splitlines():
            if line.startswith("map_Kd "):
                assert (mtl.parent / line.split(maxsplit=1)[1]).is_file()
    for obj in root.rglob("*.obj"):
        for line in obj.read_text(encoding="utf-8").splitlines():
            if line.startswith("mtllib "):
                assert (obj.parent / line.split(maxsplit=1)[1]).is_file()
    mesh_uris = [Path(str(uri.text)) for uri in world.findall(".//mesh/uri")]
    assert mesh_uris
    assert all(not uri.is_absolute() and (root / uri).is_file() for uri in mesh_uris)
    assert (root / "diagnostics" / "sdf_check.json").is_file()
    assert (root / "diagnostics" / "entity_mapping.json").is_file()
    assert (root / "diagnostics" / "runtime_preflight.json").is_file()


def test_gazebo_robot_base_mobility_follows_csd(tmp_path: Path) -> None:
    package = read_openusd_scene_package(BENCHMARK_SCENE)
    assert package.robot is not None and package.robot.fixed_base
    free_package = replace(package, robot=replace(package.robot, fixed_base=False))

    model, _ = _robot_model(tmp_path, free_package)

    assert model is not None
    assert model.find("joint[@name='robot_base_fixed']") is None


def test_gazebo_iPads_profile_compiles_from_robot_id(tmp_path: Path) -> None:
    package = read_openusd_scene_package(BENCHMARK_SCENE)
    assert package.robot is not None

    model, files = _robot_model(
        tmp_path, replace(package, robot=replace(package.robot, robot_id="ipads_desc"))
    )

    assert model is not None
    assert (
        tmp_path / "robots" / "ipads_desc" / "ipads_moveit_config" / "config" / "ipads_desc.srdf"
    ).is_file()
    assert files
    runtime = _write_runtime_artifacts(
        root=tmp_path,
        csd_id="ipads",
        urdf_source=tmp_path
        / "robots"
        / "ipads_desc"
        / "ipads_description"
        / "urdf"
        / "ipads_desc.urdf",
        camera_names=(),
    )
    assert set(runtime.trajectory_actions) == {"left_arm_group", "right_arm_group"}
    semantics = json.loads(
        (tmp_path / runtime.robot_semantics_file).read_text(encoding="utf-8")
    )
    assert semantics["controller_groups"] == {
        "left_arm_group": semantics["groups"]["left_arm_group"],
        "right_arm_group": semantics["groups"]["right_arm_group"],
    }


def test_robot_fixed_base_defaults_to_true() -> None:
    robot = _read_robot(
        """def Xform "Robot"
{
    custom string robosim:robot:id = "franka_panda"
    custom string robosim:robot:instanceId = "robot"
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}"""
    )

    assert robot is not None and robot.fixed_base


def test_gazebo_rejects_unknown_robot_profile() -> None:
    with pytest.raises(PackageError, match="profile is unavailable"):
        _robot_asset_profile("unsupported_robot")
