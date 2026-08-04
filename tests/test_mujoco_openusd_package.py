from pathlib import Path

import pytest

from robosim.core.mujoco_openusd_package import (
    OpenUsdArticulationJoint,
    _parent_to_child,
    _read_shell_materials,
    _split_visual_obj,
)


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
