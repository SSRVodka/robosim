import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import mujoco

from robosim.backends.mujoco import MuJoCoBackend
from robosim.core.mujoco_openusd_package import (
    OpenUsdArticulationJoint,
    _parent_to_child,
    _read_shell_materials,
    _split_visual_obj,
    compile_openusd_scene_package,
    read_openusd_scene_package,
)

V9_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "art_6b18395c757141bb9fa08cbcb7e6bc87"
)
V9D_SCENE = Path(__file__).resolve().parents[1] / "example" / "experiment-9d" / "scene.usda"


def test_shell_materials_read_composed_color_textures(tmp_path: Path) -> None:
    for name in ("floor.jpg", "walls.jpg"):
        (tmp_path / name).write_text("texture")
    asset = tmp_path / "asset.usda"
    asset.write_text(
        """#usda 1.0
def Xform "Asset" {
    def Xform "Visual" {
        def Xform "Floor" (prepend apiSchemas = ["MaterialBindingAPI"]) {
            rel material:binding = </Asset/FloorMaterial>
        }
        def Xform "Walls" (prepend apiSchemas = ["MaterialBindingAPI"]) {
            rel material:binding = </Asset/WallsMaterial>
        }
    }
    def Material "FloorMaterial" {
        def Shader "ColorTexture" {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @floor.jpg@
        }
    }
    def Material "WallsMaterial" {
        def Shader "ColorTexture" {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @walls.jpg@
        }
    }
}
"""
    )

    materials = _read_shell_materials(asset)

    assert [(item.name, item.texture.name) for item in materials] == [
        ("floor", "floor.jpg"),
        ("walls", "walls.jpg"),
    ]


def test_joint_frame_transform_cancels_equal_local_frames() -> None:
    joint = OpenUsdArticulationJoint(
        "joint",
        "prismatic",
        "parent",
        "child",
        "X",
        (0.156264, 0.13749123, 0.75570774),
        (0.70710677, 0.0, 0.0, -0.70710677),
        (0.156264, 0.13749123, 0.75570774),
        (0.70710677, 0.0, 0.0, -0.70710677),
        (-0.2, 0.0),
        0.0,
        0.0,
        0.0,
    )

    position, rotation = _parent_to_child(joint)

    assert position == pytest.approx((0.0, 0.0, 0.0))
    assert rotation == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_visual_obj_is_split_into_mujoco_meshes(tmp_path: Path) -> None:
    asset = tmp_path / "asset.xml"
    asset.write_text("")
    source = tmp_path / "support/obj/visual.obj"
    source.parent.mkdir(parents=True)
    source.write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
o first
f 1 2 3
o second
f 3 2 1
"""
    )

    meshes = _split_visual_obj(asset, Path("support/obj/visual.obj"))

    assert [item.as_posix() for item in meshes] == [
        "support/obj/visual_000.obj",
        "support/obj/visual_001.obj",
    ]
    assert all((tmp_path / item).is_file() for item in meshes)


def test_v9_robot_descriptor_copies_and_patches_control_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    shutil.copytree(V9_PACKAGE, package)
    scene = package / "scene.usda"
    scene.write_text(
        scene.read_text().replace(
            'def Xform "World"\n{',
            '''def Xform "World"
{
    def Xform "Robot"
    {
        custom string robosim:robot:id = "franka_panda"
        custom string robosim:robot:instanceId = "robot"
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = (1, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
    def Scope "Cameras"
    {
        def Camera "agent_view"
        {
            float focalLength = 24
            float verticalAperture = 20.25
            quatd xformOp:orient = (1, 0, 0, 0)
            double3 xformOp:translate = (4, -4, 3)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }
    }
    def Scope "Lights"
    {
        def DistantLight "key_light"
        {
            float intensity = 1
            quatd xformOp:orient = (1, 0, 0, 0)
            double3 xformOp:translate = (0, 0, 3)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }
    }''',
        )
    )
    checksum = hashlib.sha256(scene.read_bytes()).hexdigest()
    checksums = package / "checksums.sha256"
    checksums.write_text(
        "\n".join(
            f"{checksum} scene.usda" if line.endswith(" scene.usda") else line
            for line in checksums.read_text().splitlines()
        )
        + "\n"
    )
    parsed = read_openusd_scene_package(scene)
    assert parsed.robot is not None
    assert parsed.robot.robot_id == "franka_panda"
    assert parsed.robot.pose[:3] == (1.0, 2.0, 3.0)
    assert [camera.name for camera in parsed.cameras] == ["agent_view"]
    assert [light.name for light in parsed.lights] == ["key_light"]

    def preview(
        scene_path: Path, diagnostics: Path, initial_positions: dict[str, float]
    ) -> str:
        assert isinstance(initial_positions, dict)
        path = diagnostics / "preview.png"
        path.write_bytes(b"preview")
        return "diagnostics/preview.png"

    monkeypatch.setattr("robosim.core.mujoco_openusd_package._write_preview", preview)
    monkeypatch.chdir(tmp_path)
    manifest = compile_openusd_scene_package(
        csd_path=scene,
        output_root=Path("engine_manifests"),
        realization_config=None,
        realization_version="test",
        simulator_version="test",
    )
    robot_xml = Path(manifest.root_path) / "robots/franka_panda/panda.xml"
    assert robot_xml.is_file()
    root = ET.parse(robot_xml).getroot()
    assert root.find("worldbody/body").attrib["pos"] == "1.0 2.0 3.0"
    assert "robots/franka_panda/panda.srdf" in manifest.generated_files
    backend = MuJoCoBackend.from_csd_realization_manifest(manifest, headless=True)
    try:
        groups = {group.name for group in backend.get_robot_spec().joint_model_groups}
        sensors = {sensor.name for sensor in backend.list_sensors().entries}
        assert backend.robot_name == "panda"
        assert "panda_arm" in groups
        assert "agent_view" in sensors
        assert backend._model.camera("agent_view").name == "agent_view"
        assert backend._model.light("key_light").name == "key_light"
        assert backend._model.ncam == 1
        assert backend._model.nlight == 1
    finally:
        backend.shutdown()


def test_v9d_shell_realizes_separate_floor_and_walls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def preview(_scene: Path, diagnostics: Path, _positions: dict[str, float]) -> str:
        (diagnostics / "preview.png").write_bytes(b"preview")
        return "diagnostics/preview.png"

    monkeypatch.setattr(
        "robosim.core.mujoco_openusd_package._write_preview",
        preview,
    )
    manifest = compile_openusd_scene_package(
        csd_path=V9D_SCENE,
        output_root=tmp_path / "engine_manifests",
        realization_config=None,
        realization_version="test",
        simulator_version="test",
    )

    root = Path(manifest.root_path)
    shell = next(
        path
        for path in (root / "models").glob("*/asset.xml")
        if {body.attrib["name"] for body in ET.parse(path).findall("worldbody/body/body")}
        >= {"Floor", "Walls"}
    )
    assets = ET.parse(shell).find("asset")
    assert assets is not None
    assert {item.attrib["name"] for item in assets.findall("material")} >= {
        "floor_material",
        "walls_material",
    }
    wall_visuals = [
        item for item in assets.findall("mesh") if "walls/visual_" in item.attrib["file"]
    ]
    assert len(wall_visuals) == 4
    stool = next(
        path
        for path in (root / "models").glob("*/asset.xml")
        if any(
            item.attrib.get("file") == "Image_0.png"
            for item in ET.parse(path).findall("asset/texture")
        )
    )
    assert any("material" in item.attrib for item in ET.parse(stool).findall("worldbody/body/geom"))
    assert manifest.preview_files == ("diagnostics/preview.png",)


def test_composed_drive_target_initializes_mujoco_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    shutil.copytree(V9_PACKAGE, package)
    scene = package / "scene.usda"
    scene.write_text(
        scene.read_text()
        .replace("    defaultPrim", "    subLayers = [@override.usda@]\n    defaultPrim")
        .replace("float drive:angular:physics:targetPosition = 0\n", "")
    )
    override = package / "override.usda"
    override.write_text(
        '''#usda 1.0
over "World" {
    over "Objects" {
        over "cabinet_double_door_01" {
            over "Joints" {
                over "RevoluteJoint_double_door_4_down_joint" (
                    prepend apiSchemas = ["PhysicsDriveAPI:angular"]
                ) {
                    float drive:angular:physics:targetPosition = 30
                }
            }
        }
    }
}
'''
    )
    checksums = package / "checksums.sha256"
    checksums.write_text(
        "\n".join(
            (
                *(line for line in checksums.read_text().splitlines() if not line.endswith(" scene.usda")),
                f"{hashlib.sha256(scene.read_bytes()).hexdigest()} scene.usda",
                f"{hashlib.sha256(override.read_bytes()).hexdigest()} override.usda",
            )
        )
        + "\n"
    )
    monkeypatch.setattr(
        "robosim.core.mujoco_openusd_package._write_preview",
        lambda _scene, diagnostics, _positions: _preview_stub(diagnostics),
    )
    manifest = compile_openusd_scene_package(
        csd_path=scene,
        output_root=tmp_path / "engine_manifests",
        realization_config=None,
        realization_version="test",
        simulator_version="test",
    )
    state = json.loads(
        (Path(manifest.root_path) / "runtime/initial_joint_positions.json").read_text()
    )
    name = next(key for key in state if key.endswith("/RevoluteJoint_double_door_4_down_joint"))
    assert state[name] == pytest.approx(0.5235987755982988)
    backend = MuJoCoBackend.from_csd_realization_manifest(manifest, headless=True)
    try:
        joint_id = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert backend._data.qpos[backend._model.jnt_qposadr[joint_id]] == pytest.approx(state[name])
    finally:
        backend.shutdown()


def test_mujoco_realization_loads_after_relocation_in_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "robosim.core.mujoco_openusd_package._write_preview",
        lambda _scene, diagnostics, _positions: _preview_stub(diagnostics),
    )
    manifest = compile_openusd_scene_package(
        csd_path=V9_PACKAGE / "scene.usda",
        output_root=tmp_path / "engine_manifests",
        realization_config=None,
        realization_version="test",
        simulator_version="test",
    )
    relocated = tmp_path / "relocated"
    shutil.move(manifest.root_path, relocated)
    program = (
        "from robosim.backends.mujoco import MuJoCoBackend; "
        "import sys; "
        "backend = MuJoCoBackend.from_csd_realization_manifest_file(sys.argv[1], headless=True); "
        "backend.shutdown()"
    )
    completed = subprocess.run(
        (sys.executable, "-c", program, str(relocated / "manifest.json")),
        check=False,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _preview_stub(diagnostics: Path) -> str:
    (diagnostics / "preview.png").write_bytes(b"preview")
    return "diagnostics/preview.png"
