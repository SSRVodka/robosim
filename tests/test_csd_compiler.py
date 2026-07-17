"""Tests for CSD -> backend-native scene compilers."""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from shutil import copy2, copytree

import mujoco
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

from robosim.core import (
    CsdRealizationManifest,
    compile_csd,
    compile_csd_to_gazebo,
    compile_csd_to_mujoco,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"
SHARED_OPENUSD_CSD = FIXTURE_ROOT / "openusd" / "shared_tabletop" / "csd.usda"
SEMANTIC_OPENUSD_ROOT = FIXTURE_ROOT / "openusd" / "semantic"
EXPECTED_MUJOCO_PREVIEW_SIZE = 512
MUJOCO_POSITIVE_CSD_FIXTURES = (
    "franka_tabletop_single_object",
    "franka_tabletop_multi_object",
    "object_only_static_and_dynamic",
    "textured_scaled_object",
    "visual_tabletop_regions",
    "object_only_default_camera",
    "object_inertial_contact",
    "tabletop_rotated_surface_object",
    "low_gravity_static_layout",
)


def _load_registry_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _csd_fixture(name: str) -> Path:
    return SEMANTIC_OPENUSD_ROOT / name.removesuffix(".json") / "csd.usda"


def _editable_csd_fixture(name: str, root: Path) -> tuple[Path, Usd.Stage]:
    source = _csd_fixture(name)
    path = root / "csd_inputs" / name.removesuffix(".json") / "csd.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, path)
    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    return path, stage


def _save_stage(stage: Usd.Stage) -> None:
    stage.GetRootLayer().Save()


def _csd_id(path: Path) -> str:
    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    return str(stage.GetPrimAtPath("/World").GetAttribute("robosim:csd:id").Get())


def _required_element(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    assert element is not None
    return element


def _fixture_mesh_half_extents(path: Path) -> tuple[float, float, float]:
    name = path.stem
    if name in {"box", "object_box"}:
        return (0.15, 0.15, 0.15)
    if name in {"anchor", "object_anchor"}:
        return (0.1, 0.1, 0.1)
    if "tray" in name:
        return (0.08, 0.055, 0.012)
    if "marker" in name:
        return (0.018, 0.018, 0.055)
    if "can" in name:
        return (0.035, 0.035, 0.08)
    if "mug" in name:
        return (0.035, 0.035, 0.055)
    return (0.035, 0.035, 0.035)


def _write_box_mesh(path: Path) -> None:
    hx, hy, hz = _fixture_mesh_half_extents(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"v {-hx} {-hy} {-hz}",
                f"v {hx} {-hy} {-hz}",
                f"v {hx} {hy} {-hz}",
                f"v {-hx} {hy} {-hz}",
                f"v {-hx} {-hy} {hz}",
                f"v {hx} {-hy} {hz}",
                f"v {hx} {hy} {hz}",
                f"v {-hx} {hy} {hz}",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            )
        ),
        encoding="utf-8",
    )


def _write_png_1x1(path: Path) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"\x00\xff\xff\xff\xff"
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_fixture_asset_files(asset_root: Path, asset_registry: Mapping[str, object]) -> None:
    records = asset_registry.get("objects", ())
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        variants = record.get("backend_resources", ())
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            relative_path = variant.get("mesh_path") or variant.get("relative_path")
            if relative_path:
                _write_box_mesh(asset_root / str(relative_path))
            collision_mesh_path = variant.get("collision_mesh_path")
            if collision_mesh_path:
                _write_box_mesh(asset_root / str(collision_mesh_path))
            material = variant.get("material")
            if isinstance(material, Mapping) and material.get("texture_path"):
                texture_path = asset_root / str(material["texture_path"])
                _write_png_1x1(texture_path)


def _read_ppm_rgb(
    path: Path,
    *,
    width: int = EXPECTED_MUJOCO_PREVIEW_SIZE,
    height: int = EXPECTED_MUJOCO_PREVIEW_SIZE,
) -> list[tuple[int, int, int]]:
    header = f"P6\n{width} {height}\n255\n".encode()
    payload = path.read_bytes()
    assert payload.startswith(header)
    pixels = payload[len(header) :]
    assert len(pixels) == width * height * 3
    return [
        (pixels[index], pixels[index + 1], pixels[index + 2]) for index in range(0, len(pixels), 3)
    ]


def _count_pixels(
    pixels: list[tuple[int, int, int]],
    predicate: Callable[[tuple[int, int, int]], bool],
) -> int:
    return sum(1 for pixel in pixels if predicate(pixel))


def test_compile_csd_to_mujoco_consumes_composed_openusd_stage(tmp_path: Path) -> None:
    asset_registry = {
        "objects": [
            {
                "asset_id": "object_box",
                "backend_resources": [
                    {
                        "backend": "mujoco",
                        "resource_id": "mujoco_object_box",
                        "mesh_path": "objects/box.obj",
                        "resource_hash": "hash_box_obj",
                    }
                ],
            },
            {
                "asset_id": "object_anchor",
                "backend_resources": [
                    {
                        "backend": "mujoco",
                        "resource_id": "mujoco_object_anchor",
                        "mesh_path": "objects/anchor.obj",
                        "resource_hash": "hash_anchor_obj",
                    }
                ],
            },
        ]
    }
    asset_root = tmp_path / "assets"
    _write_fixture_asset_files(asset_root, asset_registry)
    robot_template = (
        Path(__file__).resolve().parents[1] / "drivers_sim/mujoco/assets/robots/franka_panda"
    )

    result = compile_csd_to_mujoco(
        csd_path=SHARED_OPENUSD_CSD,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        realization_config={"robot_template_dir": str(robot_template)},
    )

    assert result.blockers == ()
    assert result.manifest is not None
    assert result.manifest.csd_id == "csd_shared_tabletop"
    model = mujoco.MjModel.from_xml_path(
        str(Path(result.manifest.root_path) / result.manifest.entry_file)
    )
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground") == -1
    assert tuple(round(float(value), 6) for value in model.body("link0").pos) == (
        -0.45,
        0.0,
        0.0,
    )
    assert tuple(round(float(value), 6) for value in model.body("link0").quat) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert tuple(round(float(value), 6) for value in model.geom("dynamic_box_geom").size) == (
        0.15,
        0.15,
        0.15,
    )
    assert tuple(round(float(value), 6) for value in model.geom("anchor_geom").size) == (
        0.1,
        0.1,
        0.1,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=512, width=512)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="Camera")
        segmentation = renderer.render()
        for geom_name in ("dynamic_box_geom", "anchor_geom"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            visible_pixels = int(
                (
                    (segmentation[:, :, 0] == geom_id)
                    & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                ).sum()
            )
            assert visible_pixels > 100
    finally:
        renderer.close()


@pytest.mark.parametrize("fixture_name", MUJOCO_POSITIVE_CSD_FIXTURES)
def test_compile_csd_to_mujoco_loads_and_renders_positive_fixtures(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture(fixture_name)
    csd_id = _csd_id(csd_path)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / csd_id
    scene_path = scene_root / "scene.xml"
    preview_path = scene_root / "diagnostics" / "semantic_preview.ppm"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert scene_path.is_file()
    mujoco.MjModel.from_xml_path(str(scene_path))
    pixels = _read_ppm_rgb(preview_path)
    assert max(max(pixel) for pixel in pixels) > min(min(pixel) for pixel in pixels)


def test_compile_csd_to_mujoco_writes_loadable_mjcf_and_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("franka_tabletop_single_object")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)

    result = compile_csd(
        backend="mujoco",
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        realization_config={"robot_template_dir": str(template_copy)},
        realization_version="test-0.1",
        simulator_version="test-mujoco",
    )
    manifest = result.manifest

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_tabletop_0001"
    scene_path = scene_root / "scene.xml"
    manifest_path = scene_root / "manifest.json"
    copied_mesh_path = scene_root / "assets" / "objects" / "mug.obj"
    copied_robot_xml = scene_root / "assets" / "robots" / "franka_panda" / "panda.xml"
    copied_robot_srdf = scene_root / "assets" / "robots" / "franka_panda" / "panda.srdf"
    copied_robot_mesh = scene_root / "assets" / "link0.stl"
    tree = ET.parse(scene_path)
    root = tree.getroot()

    assert isinstance(manifest, CsdRealizationManifest)
    assert result.blockers == ()
    assert manifest.csd_id == "csd_tabletop_0001"
    assert manifest.backend == "mujoco"
    assert manifest.entry_file == "scene.xml"
    assert "manifest.json" in manifest.generated_files
    assert "scene.xml" in manifest.generated_files
    assert "assets/objects/mug.obj" in manifest.generated_files
    assert "assets/robots/franka_panda/panda.xml" in manifest.generated_files
    assert copied_mesh_path.is_file()
    assert copied_robot_xml.is_file()
    assert copied_robot_srdf.is_file()
    assert copied_robot_mesh.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest.to_json_dict()
    assert root.tag == "mujoco"
    assert root.find("compiler") is None
    assert _required_element(root, "visual/global").attrib == {
        "offwidth": str(EXPECTED_MUJOCO_PREVIEW_SIZE),
        "offheight": str(EXPECTED_MUJOCO_PREVIEW_SIZE),
    }
    assert _required_element(root, "include").attrib["file"] == (
        "assets/robots/franka_panda/panda.xml"
    )
    assert _required_element(root, "asset/mesh").attrib == {
        "name": "object_mug",
        "file": "objects/mug.obj",
    }
    assert _required_element(root, "worldbody/camera").attrib["name"] == "world_camera"
    body = _required_element(root, "worldbody/body[@name='mug']")
    assert body.attrib["name"] == "mug"
    assert body.attrib["pos"] == "0.25 -0.1 0.82"
    assert body.attrib["quat"] == "1 0 0 0"
    assert body.find("freejoint") is not None
    assert _required_element(body, "geom").attrib["mesh"] == "object_mug"

    (asset_root / "objects" / "mug.obj").unlink()
    copied_template_source = template_copy
    for source_file in copied_template_source.rglob("*"):
        if source_file.is_file():
            source_file.unlink()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody >= 2


def test_compile_csd_to_mujoco_reuses_complete_cached_realization(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("franka_tabletop_single_object")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)
    output_root = tmp_path / "engine_manifests"
    realization_config = {"robot_template_dir": str(template_copy)}

    first = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
        realization_config=realization_config,
        realization_version="test-0.1",
        simulator_version="test-mujoco",
    )
    assert isinstance(first.manifest, CsdRealizationManifest)

    (asset_root / "objects" / "mug.obj").unlink()
    for source_file in template_copy.rglob("*"):
        if source_file.is_file():
            source_file.unlink()

    second = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
        realization_config=realization_config,
        realization_version="test-0.1",
        simulator_version="test-mujoco",
    )

    assert second.blockers == ()
    assert second.manifest == first.manifest
    assert (output_root / "mujoco" / "csd_tabletop_0001" / "scene.xml").is_file()


def test_compile_csd_to_mujoco_rebuilds_incomplete_cached_realization(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    output_root = tmp_path / "engine_manifests"

    first = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
    )
    assert isinstance(first.manifest, CsdRealizationManifest)
    preview_path = (
        output_root / "mujoco" / "csd_object_only_0001" / "diagnostics" / "semantic_preview.ppm"
    )
    preview_path.unlink()

    second = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
    )

    assert second.blockers == ()
    assert isinstance(second.manifest, CsdRealizationManifest)
    assert preview_path.is_file()
    assert second.manifest.cache_key == first.manifest.cache_key


def test_compile_csd_to_mujoco_cache_key_includes_default_simulator_version(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    default_result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "default" / "engine_manifests",
        asset_root=asset_root,
    )
    explicit_result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "explicit" / "engine_manifests",
        asset_root=asset_root,
        simulator_version=mujoco.__version__,
    )

    assert isinstance(default_result.manifest, CsdRealizationManifest)
    assert isinstance(explicit_result.manifest, CsdRealizationManifest)
    assert default_result.manifest.cache_key == explicit_result.manifest.cache_key


def test_compile_csd_to_mujoco_handles_multi_object_static_dynamic_scene(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("franka_tabletop_multi_object")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_path = tmp_path / "engine_manifests" / "mujoco" / "csd_tabletop_multi_0001" / "scene.xml"
    relationship_check_path = (
        tmp_path
        / "engine_manifests"
        / "mujoco"
        / "csd_tabletop_multi_0001"
        / "diagnostics"
        / "relationship_check.json"
    )
    root = ET.parse(scene_path).getroot()
    bodies = {body.attrib["name"]: body for body in root.findall("worldbody/body")}

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert "diagnostics/relationship_check.json" in result.manifest.generated_files
    assert {"mug", "tray", "marker", "surface_tabletop"} <= set(bodies)
    assert bodies["mug"].find("freejoint") is not None
    assert bodies["marker"].find("freejoint") is not None
    assert bodies["tray"].find("freejoint") is None
    assert _required_element(bodies["mug"], "geom").attrib["friction"] == "0.8 0.005 0.0001"
    relationship_check = json.loads(relationship_check_path.read_text(encoding="utf-8"))
    assert relationship_check["status"] == "passed"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody >= 4


def test_compile_csd_to_mujoco_writes_load_check_diagnostics(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    load_check_path = scene_root / "diagnostics" / "load_check.json"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert "diagnostics/load_check.json" in result.manifest.generated_files

    payload = json.loads(load_check_path.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["schema_version"] == "0.1"
    assert payload["backend"] == "mujoco"
    assert payload["csd_id"] == "csd_object_only_0001"
    assert payload["entry_file"] == "scene.xml"
    assert payload["status"] == "passed"
    assert checks["model_load"]["status"] == "passed"
    assert checks["gravity"]["expected"] == [0.0, 0.0, -9.81]
    assert checks["gravity"]["actual"] == [0.0, 0.0, -9.81]
    assert checks["body_pose:tray"]["status"] == "passed"
    assert checks["body_pose:tray"]["expected"] == [0.0, 0.0, 0.15]
    assert checks["body_pose:tray"]["actual"] == [0.0, 0.0, 0.15]
    assert checks["body_pose:mug"]["status"] == "passed"
    assert checks["body_pose:mug"]["expected"] == [0.05, 0.0, 0.32]
    assert checks["body_pose:mug"]["actual"] == [0.05, 0.0, 0.32]
    assert checks["body_mass:tray"]["status"] == "passed"
    assert checks["body_mass:tray"]["expected"] == [1.0]
    assert checks["body_mass:tray"]["actual"] == [1.0]
    assert checks["body_mass:mug"]["status"] == "passed"
    assert checks["body_mass:mug"]["expected"] == [0.2]
    assert checks["body_mass:mug"]["actual"] == [0.2]
    assert checks["geom_friction:mug_geom"]["status"] == "passed"
    assert checks["geom_friction:mug_geom"]["expected"] == [0.8, 0.005, 0.0001]
    assert checks["geom_friction:mug_geom"]["actual"] == [0.8, 0.005, 0.0001]
    assert checks["camera_pose:world_camera"]["status"] == "passed"
    assert checks["camera_pose:world_camera"]["expected"] == [1.4, 0.0, 1.2]
    assert checks["camera_pose:world_camera"]["actual"] == [1.4, 0.0, 1.2]
    assert checks["camera_orientation:world_camera"]["status"] == "passed"
    assert checks["camera_orientation:world_camera"]["expected"] == [
        0.612375,
        0.35355,
        0.35355,
        0.612375,
    ]
    assert checks["camera_orientation:world_camera"]["actual"] == pytest.approx(
        [0.612375, 0.35355, 0.35355, 0.612375], abs=2e-6
    )
    assert checks["light_pose:key_light"]["status"] == "passed"
    assert checks["light_pose:key_light"]["expected"] == [0.0, -1.0, 3.0]
    assert checks["light_pose:key_light"]["actual"] == [0.0, -1.0, 3.0]
    assert checks["light_direction:key_light"]["status"] == "passed"
    assert checks["light_direction:key_light"]["expected"] == [0.0, 0.0, -1.0]
    assert checks["light_direction:key_light"]["actual"] == [0.0, 0.0, -1.0]


def test_compile_csd_to_mujoco_writes_orientation_load_diagnostics(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/Environment/surface_tabletop").GetAttribute("xformOp:orient").Set(
        Gf.Quatd(0.9238795325, 0.0, 0.0, 0.3826834324)
    )
    stage.GetPrimAtPath("/World/Objects/mug").GetAttribute("xformOp:orient").Set(
        Gf.Quatd(0.7071067812, 0.0, 0.0, 0.7071067812)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    load_check_path = (
        tmp_path
        / "engine_manifests"
        / "mujoco"
        / "csd_tabletop_0001"
        / "diagnostics"
        / "load_check.json"
    )
    payload = json.loads(load_check_path.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in payload["checks"]}

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert checks["body_orientation:mug"]["status"] == "passed"
    assert checks["body_orientation:mug"]["expected"] == [0.707107, 0.0, 0.0, 0.707107]
    assert checks["body_orientation:mug"]["actual"] == [0.707107, 0.0, 0.0, 0.707107]
    assert checks["surface_orientation:surface_tabletop"]["status"] == "passed"
    assert checks["surface_orientation:surface_tabletop"]["expected"] == [
        0.92388,
        0.0,
        0.0,
        0.382683,
    ]
    assert checks["surface_orientation:surface_tabletop"]["actual"] == [
        0.92388,
        0.0,
        0.0,
        0.382683,
    ]
    assert checks["surface_size:surface_tabletop_geom"]["status"] == "passed"
    assert checks["surface_size:surface_tabletop_geom"]["expected"] == [0.45, 0.35, 0.04]
    assert checks["surface_size:surface_tabletop_geom"]["actual"] == [0.45, 0.35, 0.04]
    assert checks["surface_friction:surface_tabletop_geom"]["status"] == "passed"
    assert checks["surface_friction:surface_tabletop_geom"]["expected"] == [1.2, 0.2, 0.2]
    assert checks["surface_friction:surface_tabletop_geom"]["actual"] == [1.2, 0.2, 0.2]
    assert checks["surface_rgba:surface_tabletop_geom"]["status"] == "passed"
    assert checks["surface_rgba:surface_tabletop_geom"]["expected"] == [0.42, 0.36, 0.28, 1.0]
    assert checks["surface_rgba:surface_tabletop_geom"]["actual"] == [0.42, 0.36, 0.28, 1.0]


def test_compile_csd_to_mujoco_writes_physics_check_diagnostics(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    physics_check_path = scene_root / "diagnostics" / "physics_check.json"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert "diagnostics/physics_check.json" in result.manifest.generated_files

    payload = json.loads(physics_check_path.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["schema_version"] == "0.1"
    assert payload["backend"] == "mujoco"
    assert payload["csd_id"] == "csd_object_only_0001"
    assert payload["status"] == "passed"
    assert checks["mj_forward"]["status"] == "passed"
    assert checks["finite_state_after_steps"]["status"] == "passed"
    assert checks["finite_state_after_steps"]["details"]["steps"] == 25


def test_compile_csd_to_mujoco_writes_validation_record(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    validation_record_path = scene_root / "diagnostics" / "validation_record.json"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert "diagnostics/validation_record.json" in result.manifest.generated_files

    payload = json.loads(validation_record_path.read_text(encoding="utf-8"))

    assert payload == {
        "backend": "mujoco",
        "cache_key": result.manifest.cache_key,
        "csd_id": "csd_object_only_0001",
        "evidence_files": [
            "diagnostics/load_check.json",
            "diagnostics/relationship_check.json",
            "diagnostics/physics_check.json",
        ],
        "manifest_id": "manifest_mujoco_csd_object_only_0001",
        "preview_files": ["diagnostics/semantic_preview.ppm"],
        "schema_version": "0.1",
        "status": "passed",
        "validation_id": "validation_mujoco_csd_object_only_0001",
    }


def test_compile_csd_to_mujoco_realizes_world_template_geometry(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    tabletop_result = compile_csd_to_mujoco(
        csd_path=_csd_fixture("franka_tabletop_single_object"),
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )
    empty_floor_result = compile_csd_to_mujoco(
        csd_path=_csd_fixture("object_only_static_and_dynamic"),
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    tabletop_root = ET.parse(
        tmp_path / "engine_manifests" / "mujoco" / "csd_tabletop_0001" / "scene.xml"
    ).getroot()
    tabletop_load_check_path = (
        tmp_path
        / "engine_manifests"
        / "mujoco"
        / "csd_tabletop_0001"
        / "diagnostics"
        / "load_check.json"
    )
    tabletop_load_check = json.loads(tabletop_load_check_path.read_text(encoding="utf-8"))
    load_checks = {check["name"]: check for check in tabletop_load_check["checks"]}
    empty_floor_root = ET.parse(
        tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001" / "scene.xml"
    ).getroot()
    table_body = _required_element(tabletop_root, "worldbody/body[@name='surface_tabletop']")
    table_geom = _required_element(table_body, "geom")

    assert tabletop_result.blockers == ()
    assert empty_floor_result.blockers == ()
    assert table_body.find("freejoint") is None
    assert table_body.attrib["pos"] == "0.5 0 0.74"
    assert table_body.attrib["quat"] == "1 0 0 0"
    assert table_geom.attrib == {
        "name": "surface_tabletop_geom",
        "type": "box",
        "size": "0.45 0.35 0.04",
        "rgba": "0.42 0.36 0.28 1",
        "friction": "1.2 0.2 0.2",
    }
    assert load_checks["surface_pose:surface_tabletop"]["status"] == "passed"
    assert load_checks["surface_pose:surface_tabletop"]["expected"] == [0.5, 0.0, 0.74]
    assert load_checks["surface_pose:surface_tabletop"]["actual"] == [0.5, 0.0, 0.74]
    assert empty_floor_root.find("worldbody/body[@name='surface_tabletop']") is None


def test_compile_csd_to_mujoco_uses_typed_environment_gravity(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute("physics:gravityMagnitude").Set(1.62)
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_path = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001" / "scene.xml"
    root = ET.parse(scene_path).getroot()

    assert result.blockers == ()
    assert _required_element(root, "option").attrib["gravity"] == "0 0 -1.62"


def test_compile_csd_to_mujoco_preserves_explicit_friction_tuple(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    stage.GetPrimAtPath("/World/Objects/mug").CreateAttribute(
        "robosim:mujoco:friction", Sdf.ValueTypeNames.Double3
    ).Set(Gf.Vec3d(0.9, 0.02, 0.003))
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_path = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001" / "scene.xml"
    root = ET.parse(scene_path).getroot()
    mug_body = _required_element(root, "worldbody/body[@name='mug']")
    geom = _required_element(mug_body, "geom")

    assert result.blockers == ()
    assert geom.attrib["friction"] == "0.9 0.02 0.003"


def test_compile_csd_to_mujoco_preserves_object_contact_parameters(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    mug = stage.GetPrimAtPath("/World/Objects/mug")
    mug.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.004)
    mug.CreateAttribute("mjc:gap", Sdf.ValueTypeNames.Double).Set(0.001)
    mug.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.02, 1.0])
    mug.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set(
        [0.9, 0.95, 0.001, 0.5, 2.0]
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_path = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001" / "scene.xml"
    root = ET.parse(scene_path).getroot()
    mug_body = _required_element(root, "worldbody/body[@name='mug']")
    geom = _required_element(mug_body, "geom")

    assert result.blockers == ()
    assert geom.attrib["margin"] == "0.004"
    assert geom.attrib["gap"] == "0.001"
    assert geom.attrib["solref"] == "0.02 1"
    assert geom.attrib["solimp"] == "0.9 0.95 0.001 0.5 2"

    load_check = json.loads(
        (
            tmp_path
            / "engine_manifests"
            / "mujoco"
            / "csd_object_only_0001"
            / "diagnostics"
            / "load_check.json"
        ).read_text(encoding="utf-8")
    )
    checks = {check["name"]: check for check in load_check["checks"]}

    assert checks["geom_contact:mug_geom"]["status"] == "passed"
    assert checks["geom_contact:mug_geom"]["expected"] == {
        "gap": [0.001],
        "margin": [0.004],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "solref": [0.02, 1.0],
    }
    assert checks["geom_contact:mug_geom"]["actual"] == {
        "gap": [0.001],
        "margin": [0.004],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "solref": [0.02, 1.0],
    }


def test_compile_csd_to_mujoco_preserves_explicit_object_inertial(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    mug = stage.GetPrimAtPath("/World/Objects/mug")
    mug.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(0.01, 0.0, 0.02)
    )
    mug.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(0.002, 0.003, 0.004)
    )
    mug.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    scene_path = scene_root / "scene.xml"
    root = ET.parse(scene_path).getroot()
    mug_body = _required_element(root, "worldbody/body[@name='mug']")
    inertial = _required_element(mug_body, "inertial")
    geom = _required_element(mug_body, "geom")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    loaded_mug = model.body("mug")
    load_check = json.loads((scene_root / "diagnostics" / "load_check.json").read_text())
    checks = {check["name"]: check for check in load_check["checks"]}

    assert result.blockers == ()
    assert inertial.attrib == {
        "pos": "0.01 0 0.02",
        "mass": "0.2",
        "diaginertia": "0.002 0.003 0.004",
    }
    assert geom.attrib["density"] == "0"
    assert "mass" not in geom.attrib
    assert tuple(round(float(value), 6) for value in loaded_mug.ipos) == (0.01, 0.0, 0.02)
    assert tuple(round(float(value), 6) for value in loaded_mug.inertia) == (
        0.002,
        0.003,
        0.004,
    )
    assert checks["body_inertial_pos:mug"]["status"] == "passed"
    assert checks["body_inertia:mug"]["status"] == "passed"


def test_compile_csd_to_mujoco_blocks_invalid_physical_parameters(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    stage.GetPrimAtPath("/World/Objects/tray").GetAttribute("physics:mass").Set(0.0)
    mug = stage.GetPrimAtPath("/World/Objects/mug")
    mug.CreateAttribute("robosim:mujoco:friction", Sdf.ValueTypeNames.Double3).Set(
        Gf.Vec3d(-0.1, 0.005, 0.0001)
    )
    mug.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.001)
    mug.CreateAttribute("mjc:gap", Sdf.ValueTypeNames.Double).Set(0.002)
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert [blocker.to_json_dict() for blocker in result.blockers] == [
        {
            "blocker_id": "csd_object_only_0001_mujoco_tray_mass_kg_compile_blocked",
            "csd_id": "csd_object_only_0001",
            "backend": "mujoco",
            "asset_id": "tray",
            "scope": "csd",
            "reason": "object tray mass_kg must be positive",
        },
        {
            "blocker_id": "csd_object_only_0001_mujoco_mug_friction_compile_blocked",
            "csd_id": "csd_object_only_0001",
            "backend": "mujoco",
            "asset_id": "mug",
            "scope": "csd",
            "reason": "object mug friction values must be non-negative",
        },
        {
            "blocker_id": "csd_object_only_0001_mujoco_mug_contact_compile_blocked",
            "csd_id": "csd_object_only_0001",
            "backend": "mujoco",
            "asset_id": "mug",
            "scope": "csd",
            "reason": "object mug contact gap_m must be less than or equal to margin_m",
        },
    ]


def test_compile_csd_to_mujoco_blocks_invalid_inertial_parameters(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    mug = stage.GetPrimAtPath("/World/Objects/mug")
    mug.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    mug.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(0.002, 0.0, 0.004)
    )
    mug.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_mug_inertial_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "mug",
        "scope": "csd",
        "reason": ("object mug diagonal_inertia_kg_m2 values must be positive and finite"),
    }


def test_compile_csd_to_mujoco_preserves_texture_material_and_mesh_scale(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("textured_scaled_object")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_textured_scaled_0001"
    scene_path = scene_root / "scene.xml"
    root = ET.parse(scene_path).getroot()
    mesh = _required_element(root, "asset/mesh")
    texture = _required_element(root, "asset/texture")
    material = _required_element(root, "asset/material")
    geom = _required_element(root, "worldbody/body/geom")

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert mesh.attrib["scale"] == "0.75 0.75 0.75"
    assert texture.attrib == {
        "name": "can_label_texture",
        "type": "2d",
        "file": "textures/can_label.png",
    }
    assert material.attrib == {
        "name": "can_label",
        "texture": "can_label_texture",
        "rgba": "1 1 1 1",
    }
    assert geom.attrib["material"] == "can_label"
    assert (scene_root / "assets" / "textures" / "can_label.png").is_file()
    assert "assets/textures/can_label.png" in result.manifest.generated_files

    (asset_root / "objects" / "can.obj").unlink()
    (asset_root / "textures" / "can_label.png").unlink()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.ngeom == 1


def test_compile_csd_to_mujoco_preserves_separate_collision_mesh(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    stage.GetPrimAtPath("/World/Objects/mug").CreateAttribute(
        "mjc:margin", Sdf.ValueTypeNames.Double
    ).Set(0.003)
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    records = asset_registry["objects"]
    assert isinstance(records, list)
    for record in records:
        if isinstance(record, dict) and record.get("asset_id") == "object_mug":
            resources = record["backend_resources"]
            assert isinstance(resources, list)
            resource = resources[0]
            assert isinstance(resource, dict)
            resource["collision_mesh_path"] = "collision/mug_collision.obj"
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    scene_path = scene_root / "scene.xml"
    root = ET.parse(scene_path).getroot()
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall("asset/mesh")}
    mug_body = _required_element(root, "worldbody/body[@name='mug']")
    visual_geom = _required_element(mug_body, "geom[@name='mug_geom']")
    collision_geom = _required_element(mug_body, "geom[@name='mug_collision_geom']")
    load_check = json.loads(
        (scene_root / "diagnostics" / "load_check.json").read_text(encoding="utf-8")
    )
    checks = {check["name"]: check for check in load_check["checks"]}

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert meshes["object_mug"].attrib["file"] == "objects/mug.obj"
    assert meshes["object_mug_collision"].attrib["file"] == "collision/mug_collision.obj"
    assert visual_geom.attrib["mesh"] == "object_mug"
    assert visual_geom.attrib["contype"] == "0"
    assert visual_geom.attrib["conaffinity"] == "0"
    assert "mass" not in visual_geom.attrib
    assert "margin" not in visual_geom.attrib
    assert collision_geom.attrib["mesh"] == "object_mug_collision"
    assert collision_geom.attrib["mass"] == "0.2"
    assert collision_geom.attrib["margin"] == "0.003"
    assert collision_geom.attrib["rgba"] == "0 0 0 0"
    assert checks["geom_friction:mug_collision_geom"]["status"] == "passed"
    assert checks["geom_contact:mug_collision_geom"]["actual"]["margin"] == [0.003]
    assert "assets/collision/mug_collision.obj" in result.manifest.generated_files

    (asset_root / "objects" / "mug.obj").unlink()
    (asset_root / "collision" / "mug_collision.obj").unlink()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.ngeom == 3


def test_compile_csd_to_mujoco_renders_semantic_preview_screenshot(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001"
    scene_path = scene_root / "scene.xml"
    screenshot_path = scene_root / "diagnostics" / "semantic_preview.ppm"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    tray_body = model.body("tray")
    mug_body = model.body("mug")
    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert result.manifest.preview_files == ("diagnostics/semantic_preview.ppm",)
    assert tuple(round(float(value), 6) for value in tray_body.pos) == (0.0, 0.0, 0.15)
    assert tuple(round(float(value), 6) for value in mug_body.pos) == (0.05, 0.0, 0.32)

    payload = screenshot_path.read_bytes()

    header = f"P6\n{EXPECTED_MUJOCO_PREVIEW_SIZE} {EXPECTED_MUJOCO_PREVIEW_SIZE}\n255\n".encode()
    assert payload.startswith(header)
    assert screenshot_path.stat().st_size > EXPECTED_MUJOCO_PREVIEW_SIZE**2
    pixels = payload[len(header) :]
    assert max(pixels) > min(pixels)


def test_compile_csd_to_mujoco_preview_contains_distinct_visual_regions(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("visual_tabletop_regions")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_root = tmp_path / "engine_manifests" / "mujoco" / "csd_visual_tabletop_regions_0001"
    scene_path = scene_root / "scene.xml"
    screenshot_path = scene_root / "diagnostics" / "semantic_preview.ppm"
    load_check = json.loads(
        (scene_root / "diagnostics" / "load_check.json").read_text(encoding="utf-8")
    )
    checks = {check["name"]: check for check in load_check["checks"]}
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    pixels = _read_ppm_rgb(screenshot_path)

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)
    assert result.manifest.preview_files == ("diagnostics/semantic_preview.ppm",)
    assert checks["surface_rgba:surface_green_base_geom"]["status"] == "passed"
    assert checks["surface_rgba:surface_red_left_pad_geom"]["status"] == "passed"
    assert checks["surface_rgba:surface_blue_right_pad_geom"]["status"] == "passed"
    assert tuple(round(float(value), 6) for value in model.body("mug_left").pos) == (
        0.28,
        -0.2,
        0.18,
    )
    assert tuple(round(float(value), 6) for value in model.body("tray_right").pos) == (
        0.32,
        0.2,
        0.14,
    )
    assert _count_pixels(pixels, lambda pixel: pixel[0] > pixel[1] + 20) > 40
    assert _count_pixels(pixels, lambda pixel: pixel[2] > pixel[1] + 20) > 40
    assert _count_pixels(pixels, lambda pixel: pixel[1] > pixel[0] + 20) > 40


def test_compile_csd_to_mujoco_reports_missing_resource_adapter(tmp_path: Path) -> None:
    result = compile_csd_to_mujoco(
        csd_path=SHARED_OPENUSD_CSD,
        asset_registry={"objects": []},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_shared_tabletop_mujoco_object_box_resource_missing",
        "csd_id": "csd_shared_tabletop",
        "backend": "mujoco",
        "asset_id": "object_box",
        "scope": "asset",
        "reason": "asset has no backend resource adapter for mujoco",
    }


def test_compile_csd_to_mujoco_blocks_unsupported_mesh_format(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    records = asset_registry["objects"]
    assert isinstance(records, list)
    for record in records:
        if isinstance(record, dict) and record.get("asset_id") == "object_mug":
            resources = record["backend_resources"]
            assert isinstance(resources, list)
            resource = resources[0]
            assert isinstance(resource, dict)
            resource["mesh_path"] = "objects/mug.dae"
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_object_mug_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "MuJoCo mesh resource format is unsupported: objects/mug.dae",
    }


def test_compile_csd_to_mujoco_blocks_mesh_path_traversal(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    records = asset_registry["objects"]
    assert isinstance(records, list)
    for record in records:
        if isinstance(record, dict) and record.get("asset_id") == "object_mug":
            resources = record["backend_resources"]
            assert isinstance(resources, list)
            resource = resources[0]
            assert isinstance(resource, dict)
            resource["mesh_path"] = "../objects/mug.obj"
    _write_box_mesh(asset_root / "objects" / "tray.obj")

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_object_mug_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "backend resource path must stay inside asset root: ../objects/mug.obj",
    }


def test_compile_csd_to_mujoco_blocks_unsupported_collision_mesh_format(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("object_only_static_and_dynamic")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    records = asset_registry["objects"]
    assert isinstance(records, list)
    for record in records:
        if isinstance(record, dict) and record.get("asset_id") == "object_mug":
            resources = record["backend_resources"]
            assert isinstance(resources, list)
            resource = resources[0]
            assert isinstance(resource, dict)
            resource["collision_mesh_path"] = "collision/mug_collision.ply"
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_object_mug_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": (
            "MuJoCo collision mesh resource format is unsupported: collision/mug_collision.ply"
        ),
    }


def test_compile_csd_to_mujoco_blocks_texture_path_traversal(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("textured_scaled_object")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    records = asset_registry["objects"]
    assert isinstance(records, list)
    for record in records:
        if isinstance(record, dict) and record.get("asset_id") == "object_textured_can":
            resources = record["backend_resources"]
            assert isinstance(resources, list)
            resource = resources[0]
            assert isinstance(resource, dict)
            material = resource["material"]
            assert isinstance(material, dict)
            material["texture_path"] = "../textures/can_label.png"
    _write_box_mesh(asset_root / "objects" / "can.obj")

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_textured_scaled_0001_mujoco_object_textured_can_compile_blocked",
        "csd_id": "csd_textured_scaled_0001",
        "backend": "mujoco",
        "asset_id": "object_textured_can",
        "scope": "asset",
        "reason": (
            "asset material texture path must stay inside asset root: ../textures/can_label.png"
        ),
    }


def test_compile_csd_to_mujoco_reports_unsupported_robot_asset(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path = _csd_fixture("unsupported_robot")
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_unknown_robot_mujoco_robot_unknown_compile_blocked",
        "csd_id": "csd_unknown_robot",
        "backend": "mujoco",
        "asset_id": "robot_unknown",
        "scope": "asset",
        "reason": "no MuJoCo robot template is configured for robot asset",
    }


def test_compile_csd_to_mujoco_blocks_non_meter_stage_units(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert [blocker.to_json_dict() for blocker in result.blockers] == [
        {
            "blocker_id": "csd_object_only_0001_mujoco_scenario_units_compile_blocked",
            "csd_id": "csd_object_only_0001",
            "backend": "mujoco",
            "asset_id": "scenario_units",
            "scope": "csd",
            "reason": "MuJoCo compiler supports only CSD units='m', got '0.01m'",
        },
    ]


def test_compile_csd_to_mujoco_blocks_unsupported_environment_surface(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/Environment/surface_tabletop").SetTypeName("Cylinder")
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_tabletop_0001_mujoco_surface_tabletop_compile_blocked",
        "csd_id": "csd_tabletop_0001",
        "backend": "mujoco",
        "asset_id": "surface_tabletop",
        "scope": "csd",
        "reason": "MuJoCo compiler does not support environment surface type 'cylinder'",
    }


def test_compile_csd_to_mujoco_blocks_zero_surface_quaternion(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/Environment/surface_tabletop").GetAttribute("xformOp:orient").Set(
        Gf.Quatd(0.0, 0.0, 0.0, 0.0)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].asset_id == "csd_stage"
    assert "invalid_xform_orientation" in result.blockers[0].reason


def test_compile_csd_to_mujoco_blocks_invalid_surface_size(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/Environment/surface_tabletop").GetAttribute("xformOp:scale").Set(
        Gf.Vec3f(0.45, 0.35, -0.04)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_tabletop_0001_mujoco_surface_tabletop_compile_blocked",
        "csd_id": "csd_tabletop_0001",
        "backend": "mujoco",
        "asset_id": "surface_tabletop",
        "scope": "csd",
        "reason": "surface surface_tabletop box size values must be positive and finite",
    }


def test_compile_csd_to_mujoco_blocks_zero_object_quaternion(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("object_only_static_and_dynamic", tmp_path)
    stage.GetPrimAtPath("/World/Objects/mug").GetAttribute("xformOp:orient").Set(
        Gf.Quatd(0.0, 0.0, 0.0, 0.0)
    )
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].asset_id == "csd_stage"
    assert "invalid_xform_orientation" in result.blockers[0].reason


def test_compile_csd_to_mujoco_blocks_unknown_relationship_entity(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/Relationships/rel_mug_on_table").GetRelationship(
        "robosim:relationship:subject"
    ).SetTargets([Sdf.Path("/World/Objects/ghost")])
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].asset_id == "csd_stage"
    assert "unresolved_relationship_target" in result.blockers[0].reason


def test_compile_csd_to_mujoco_blocks_avoid_contact_violation(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_multi_object", tmp_path)
    mug = stage.GetPrimAtPath("/World/Objects/mug")
    marker = stage.GetPrimAtPath("/World/Objects/marker")
    marker.GetAttribute("xformOp:translate").Set(mug.GetAttribute("xformOp:translate").Get())
    marker.GetAttribute("xformOp:orient").Set(mug.GetAttribute("xformOp:orient").Get())
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    relationship_check = (
        tmp_path / "mujoco" / "csd_tabletop_multi_0001" / "diagnostics" / "relationship_check.json"
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_tabletop_multi_0001_mujoco_rel_mug_avoid_marker_compile_blocked",
        "csd_id": "csd_tabletop_multi_0001",
        "backend": "mujoco",
        "asset_id": "rel_mug_avoid_marker",
        "scope": "csd",
        "reason": ("MuJoCo initial state violates avoid_contact relationship rel_mug_avoid_marker"),
    }
    payload = json.loads(relationship_check.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "avoid_contact:rel_mug_avoid_marker"


def test_compile_csd_to_mujoco_blocks_on_top_of_surface_violation(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("visual_tabletop_regions", tmp_path)
    mug = stage.GetPrimAtPath("/World/Objects/mug_left")
    position = mug.GetAttribute("xformOp:translate").Get()
    mug.GetAttribute("xformOp:translate").Set(Gf.Vec3d(position[0], 0.45, position[2]))
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    relationship_check = (
        tmp_path
        / "mujoco"
        / "csd_visual_tabletop_regions_0001"
        / "diagnostics"
        / "relationship_check.json"
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": (
            "csd_visual_tabletop_regions_0001_mujoco_rel_mug_on_red_pad_compile_blocked"
        ),
        "csd_id": "csd_visual_tabletop_regions_0001",
        "backend": "mujoco",
        "asset_id": "rel_mug_on_red_pad",
        "scope": "csd",
        "reason": ("MuJoCo initial state violates on_top_of relationship rel_mug_on_red_pad"),
    }
    payload = json.loads(relationship_check.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "on_top_of:rel_mug_on_red_pad"


def test_compile_csd_to_mujoco_applies_template_nondefault_gravity(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd_path, stage = _editable_csd_fixture("franka_tabletop_single_object", tmp_path)
    stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute("physics:gravityMagnitude").Set(1.62)
    _save_stage(stage)
    asset_registry = _load_registry_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd_path=csd_path,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    scene_root = tmp_path / "mujoco" / "csd_tabletop_0001"
    scene_path = scene_root / "scene.xml"
    copied_template = scene_root / "assets" / "robots" / "franka_panda" / "panda.xml"

    assert result.blockers == ()
    assert isinstance(result.manifest, CsdRealizationManifest)

    template_root = ET.parse(copied_template).getroot()
    option = _required_element(template_root, "option")
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    assert option.attrib["gravity"] == "0 0 -1.62"
    assert tuple(round(float(value), 6) for value in model.opt.gravity) == (0.0, 0.0, -1.62)


def test_compile_csd_to_gazebo_consumes_composed_openusd_stage(tmp_path: Path) -> None:
    registry = {
        "objects": [
            {
                "asset_id": asset_id,
                "backend_resources": [
                    {
                        "backend": "gazebo",
                        "resource_id": f"gazebo_{asset_id}",
                        "mesh_path": f"objects/{asset_id}.obj",
                        "resource_hash": f"hash_{asset_id}",
                    }
                ],
            }
            for asset_id in ("object_box", "object_anchor")
        ]
    }
    asset_root = tmp_path / "assets"
    _write_fixture_asset_files(asset_root, registry)

    result = compile_csd(
        backend="gazebo",
        csd_path=SHARED_OPENUSD_CSD,
        asset_registry=registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        simulator_version="test-gazebo",
    )

    assert result.blockers == ()
    assert result.manifest is not None
    world_root = Path(result.manifest.root_path)
    root = ET.parse(world_root / "world.sdf").getroot()
    world = _required_element(root, "world")
    models = {model.attrib["name"]: model for model in world.findall("model")}

    assert root.attrib["version"] == "1.7"
    assert _required_element(world, "gravity").text == "0 0 -9.81"
    assert _required_element(world, "physics/max_step_size").text == "0.001"
    assert {"dynamic_box", "anchor", "table", "panda"} <= set(models)
    assert _required_element(models["dynamic_box"], "pose").text == "0 0 0.35 0 0 0"
    assert _required_element(models["dynamic_box"], "link/inertial/mass").text == "1"
    assert _required_element(models["dynamic_box"], "link/inertial/inertia/ixx").text == ("0.015")
    assert (
        _required_element(models["dynamic_box"], "link/collision/geometry/mesh/uri").text
        == "assets/objects/object_box.obj"
    )
    assert (
        _required_element(models["dynamic_box"], "link/visual/material/diffuse").text
        == "0.8 0.25 0.15 1"
    )
    assert _required_element(models["table"], "link/collision/geometry/box/size").text == (
        "1.2 0.8 0.1"
    )
    assert _required_element(world, "model[@name='csd_sensors']/link/sensor").attrib == {
        "name": "Camera",
        "type": "camera",
    }
    assert _required_element(world, "light").attrib["name"] == "KeyLight"
    mesh_uris = [Path(str(uri.text)) for uri in world.findall(".//mesh/uri")]
    assert mesh_uris
    assert all(not uri.is_absolute() and (world_root / uri).is_file() for uri in mesh_uris)
    assert "assets/objects/object_box.obj" in result.manifest.generated_files
    assert any(
        path.startswith("assets/robots/franka_panda/meshes/collision/")
        for path in result.manifest.generated_files
    )
    assert (world_root / "diagnostics" / "sdf_check.json").is_file()
    assert (world_root / "diagnostics" / "headless_load.json").is_file()
    assert (world_root / "diagnostics" / "validation_record.json").is_file()


def test_compile_csd_to_gazebo_reports_missing_resource_adapter(tmp_path: Path) -> None:
    result = compile_csd_to_gazebo(
        csd_path=SHARED_OPENUSD_CSD,
        asset_registry={"objects": []},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_shared_tabletop_gazebo_object_box_resource_missing",
        "csd_id": "csd_shared_tabletop",
        "backend": "gazebo",
        "asset_id": "object_box",
        "scope": "asset",
        "reason": "asset has no backend resource adapter for gazebo",
    }
