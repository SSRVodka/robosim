"""Tests for CSD -> backend-native scene compilers."""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Mapping
from pathlib import Path
from shutil import copytree

import mujoco

from robosim.core import (
    ConcreteScenarioDefinition,
    CsdObjectContact,
    CsdRealizationManifest,
    compile_csd,
    compile_csd_to_gazebo,
    compile_csd_to_mujoco,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"


def _load_json_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _required_element(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    assert element is not None
    return element


def _write_tetra_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 0.04 0 0",
                "v 0 0.04 0",
                "v 0 0 0.04",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
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
                _write_tetra_mesh(asset_root / str(relative_path))
            collision_mesh_path = variant.get("collision_mesh_path")
            if collision_mesh_path:
                _write_tetra_mesh(asset_root / str(collision_mesh_path))
            material = variant.get("material")
            if isinstance(material, Mapping) and material.get("texture_path"):
                texture_path = asset_root / str(material["texture_path"])
                _write_png_1x1(texture_path)


def test_compile_csd_to_mujoco_writes_loadable_mjcf_and_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)

    result = compile_csd(
        backend="mujoco",
        csd=csd,
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
    copied_mesh_path = (
        scene_root / "assets" / "objects" / "mug.obj"
    )
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
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)
    output_root = tmp_path / "engine_manifests"
    realization_config = {"robot_template_dir": str(template_copy)}

    first = compile_csd_to_mujoco(
        csd=csd,
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
        csd=csd,
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    output_root = tmp_path / "engine_manifests"

    first = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
    )
    assert isinstance(first.manifest, CsdRealizationManifest)
    preview_path = (
        output_root
        / "mujoco"
        / "csd_object_only_0001"
        / "diagnostics"
        / "semantic_preview.ppm"
    )
    preview_path.unlink()

    second = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=output_root,
        asset_root=asset_root,
    )

    assert second.blockers == ()
    assert isinstance(second.manifest, CsdRealizationManifest)
    assert preview_path.is_file()
    assert second.manifest.cache_key == first.manifest.cache_key


def test_compile_csd_to_mujoco_handles_multi_object_static_dynamic_scene(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_multi_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    assert checks["camera_orientation:world_camera"]["actual"] == [
        0.612375,
        0.35355,
        0.35355,
        0.612375,
    ]
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
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    objects = scenario["objects"]
    assert isinstance(environment, dict)
    surfaces = environment["surfaces"]
    assert isinstance(surfaces, list)
    assert isinstance(objects, list)
    surface = surfaces[0]
    mug = objects[0]
    assert isinstance(surface, dict)
    assert isinstance(mug, dict)
    surface_pose = surface["pose"]
    mug_pose = mug["pose"]
    assert isinstance(surface_pose, dict)
    assert isinstance(mug_pose, dict)
    surface_pose["orientation"] = {"w": 0.9238795325, "x": 0.0, "y": 0.0, "z": 0.3826834324}
    mug_pose["orientation"] = {"w": 0.7071067812, "x": 0.0, "y": 0.0, "z": 0.7071067812}
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    tabletop_result = compile_csd_to_mujoco(
        csd=_load_json_fixture("franka_tabletop_single_object.json"),
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )
    empty_floor_result = compile_csd_to_mujoco(
        csd=_load_json_fixture("object_only_static_and_dynamic.json"),
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    environment["gravity"] = [0.0, 0.0, -1.62]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )

    scene_path = tmp_path / "engine_manifests" / "mujoco" / "csd_object_only_0001" / "scene.xml"
    root = ET.parse(scene_path).getroot()

    assert result.blockers == ()
    assert _required_element(root, "option").attrib["gravity"] == "0 0 -1.62"


def test_csd_parser_exposes_typed_object_physical_state() -> None:
    csd = ConcreteScenarioDefinition.from_mapping(
        _load_json_fixture("object_only_static_and_dynamic.json")
    )
    mug = next(obj for obj in csd.objects if obj.name == "mug")

    assert mug.initial_state.mass_kg == 0.2
    assert mug.initial_state.friction == (0.8, 0.005, 0.0001)
    assert mug.initial_state.contact is None


def test_csd_parser_exposes_typed_object_contact_parameters() -> None:
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[1]
    assert isinstance(mug, dict)
    initial_state = mug["initial_state"]
    assert isinstance(initial_state, dict)
    initial_state["contact"] = {
        "margin_m": 0.004,
        "gap_m": 0.001,
        "solref": [0.02, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }

    typed = ConcreteScenarioDefinition.from_mapping(csd)
    typed_mug = next(obj for obj in typed.objects if obj.name == "mug")

    assert typed_mug.initial_state.contact == CsdObjectContact(
        margin_m=0.004,
        gap_m=0.001,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    )


def test_compile_csd_to_mujoco_preserves_explicit_friction_tuple(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[1]
    assert isinstance(mug, dict)
    initial_state = mug["initial_state"]
    assert isinstance(initial_state, dict)
    initial_state["friction"] = [0.9, 0.02, 0.003]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[1]
    assert isinstance(mug, dict)
    initial_state = mug["initial_state"]
    assert isinstance(initial_state, dict)
    initial_state["contact"] = {
        "margin_m": 0.004,
        "gap_m": 0.001,
        "solref": [0.02, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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


def test_compile_csd_to_mujoco_blocks_invalid_physical_parameters(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    tray = objects[0]
    mug = objects[1]
    assert isinstance(tray, dict)
    assert isinstance(mug, dict)
    tray_state = tray["initial_state"]
    mug_state = mug["initial_state"]
    assert isinstance(tray_state, dict)
    assert isinstance(mug_state, dict)
    tray_state["mass_kg"] = 0.0
    mug_state["friction"] = [-0.1, 0.005, 0.0001]
    mug_state["contact"] = {"margin_m": 0.001, "gap_m": 0.002}
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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


def test_compile_csd_to_mujoco_preserves_texture_material_and_mesh_scale(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("textured_scaled_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    assert model.ngeom >= 2


def test_compile_csd_to_mujoco_preserves_separate_collision_mesh(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[1]
    assert isinstance(mug, dict)
    initial_state = mug["initial_state"]
    assert isinstance(initial_state, dict)
    initial_state["contact"] = {"margin_m": 0.003}
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
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
        csd=csd,
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
    assert model.ngeom >= 4


def test_compile_csd_to_mujoco_renders_semantic_preview_screenshot(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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

    header = b"P6\n128 128\n255\n"
    assert payload.startswith(header)
    assert screenshot_path.stat().st_size > 128 * 128
    pixels = payload[len(header):]
    assert max(pixels) > min(pixels)


def test_compile_csd_to_mujoco_reports_missing_resource_adapter(tmp_path: Path) -> None:
    result = compile_csd_to_mujoco(
        csd={
            "csd_id": "csd_missing",
            "objects": [{"name": "mug", "asset_id": "object_mug"}],
        },
        asset_registry={"objects": [{"object_id": "object_mug", "variants": []}]},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_missing_mujoco_object_mug_resource_missing",
        "csd_id": "csd_missing",
        "backend": "mujoco",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "asset has no backend resource adapter for mujoco",
    }


def test_compile_csd_to_mujoco_blocks_unsupported_mesh_format(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
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
        csd=csd,
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


def test_compile_csd_to_mujoco_blocks_unsupported_collision_mesh_format(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
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
        csd=csd,
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
            "MuJoCo collision mesh resource format is unsupported: "
            "collision/mug_collision.ply"
        ),
    }


def test_compile_csd_to_mujoco_reports_unsupported_robot_asset(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("unsupported_robot.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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


def test_compile_csd_to_mujoco_blocks_unsupported_units_and_frame(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    scenario["units"] = "cm"
    scenario["frame"] = "map"
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
            "reason": "MuJoCo compiler supports only CSD units='m', got 'cm'",
        },
        {
            "blocker_id": "csd_object_only_0001_mujoco_scenario_frame_compile_blocked",
            "csd_id": "csd_object_only_0001",
            "backend": "mujoco",
            "asset_id": "scenario_frame",
            "scope": "csd",
            "reason": "MuJoCo compiler supports only frame='world', got 'map'",
        },
    ]


def test_compile_csd_to_mujoco_blocks_unsupported_environment_surface(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    surfaces = environment["surfaces"]
    assert isinstance(surfaces, list)
    surface = surfaces[0]
    assert isinstance(surface, dict)
    surface["type"] = "cylinder"
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    surfaces = environment["surfaces"]
    assert isinstance(surfaces, list)
    surface = surfaces[0]
    assert isinstance(surface, dict)
    pose = surface["pose"]
    assert isinstance(pose, dict)
    pose["orientation"] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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
        "reason": "surface surface_tabletop orientation quaternion must be non-zero",
    }


def test_compile_csd_to_mujoco_blocks_zero_object_quaternion(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[1]
    assert isinstance(mug, dict)
    pose = mug["pose"]
    assert isinstance(pose, dict)
    pose["orientation"] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_mug_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "mug",
        "scope": "csd",
        "reason": "object mug orientation quaternion must be non-zero",
    }


def test_compile_csd_to_mujoco_blocks_invalid_camera_xyaxes(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    cameras = environment["cameras"]
    assert isinstance(cameras, list)
    camera = cameras[0]
    assert isinstance(camera, dict)
    pose = camera["pose"]
    assert isinstance(pose, dict)
    pose["xyaxes"] = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_world_camera_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "world_camera",
        "scope": "csd",
        "reason": "camera world_camera xyaxes must contain non-zero non-parallel axes",
    }


def test_compile_csd_to_mujoco_blocks_zero_light_direction(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("object_only_static_and_dynamic.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    lighting = environment["lighting"]
    assert isinstance(lighting, list)
    light = lighting[0]
    assert isinstance(light, dict)
    light["direction"] = [0.0, 0.0, 0.0]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_object_only_0001_mujoco_key_light_compile_blocked",
        "csd_id": "csd_object_only_0001",
        "backend": "mujoco",
        "asset_id": "key_light",
        "scope": "csd",
        "reason": "light key_light direction must be non-zero",
    }


def test_compile_csd_to_mujoco_blocks_unknown_relationship_entity(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    relationships = scenario["relationships"]
    assert isinstance(relationships, list)
    relationship = relationships[0]
    assert isinstance(relationship, dict)
    relationship["subject"] = "object:ghost"
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_tabletop_0001_mujoco_rel_mug_on_table_compile_blocked",
        "csd_id": "csd_tabletop_0001",
        "backend": "mujoco",
        "asset_id": "rel_mug_on_table",
        "scope": "csd",
        "reason": (
            "relationship rel_mug_on_table subject references unknown entity: object:ghost"
        ),
    }


def test_compile_csd_to_mujoco_blocks_avoid_contact_violation(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_multi_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    objects = scenario["objects"]
    assert isinstance(objects, list)
    mug = objects[0]
    marker = objects[2]
    assert isinstance(mug, dict)
    assert isinstance(marker, dict)
    marker["pose"] = mug["pose"]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path,
        asset_root=asset_root,
    )

    relationship_check = (
        tmp_path
        / "mujoco"
        / "csd_tabletop_multi_0001"
        / "diagnostics"
        / "relationship_check.json"
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_tabletop_multi_0001_mujoco_rel_mug_avoid_marker_compile_blocked",
        "csd_id": "csd_tabletop_multi_0001",
        "backend": "mujoco",
        "asset_id": "rel_mug_avoid_marker",
        "scope": "csd",
        "reason": (
            "MuJoCo initial state violates avoid_contact relationship "
            "rel_mug_avoid_marker"
        ),
    }
    payload = json.loads(relationship_check.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["checks"][0]["name"] == "avoid_contact:rel_mug_avoid_marker"


def test_compile_csd_to_mujoco_applies_template_nondefault_gravity(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    scenario = csd["scenario"]
    assert isinstance(scenario, dict)
    environment = scenario["environment"]
    assert isinstance(environment, dict)
    environment["gravity"] = [0.0, 0.0, -1.62]
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd_to_mujoco(
        csd=csd,
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


def test_compile_csd_to_gazebo_writes_self_contained_sdf_world(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_gazebo.json")
    _write_fixture_asset_files(asset_root, asset_registry)

    result = compile_csd(
        backend="gazebo",
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "compiled",
        asset_root=asset_root,
        realization_version="test-0.1",
        simulator_version="test-gazebo",
    )

    manifest = result.manifest
    world_root = tmp_path / "compiled" / "gazebo" / "csd_tabletop_0001"
    world_path = world_root / "world.sdf"
    manifest_path = world_root / "manifest.json"
    copied_mesh_path = world_root / "assets" / "objects" / "mug.obj"
    root = ET.parse(world_path).getroot()
    model = _required_element(root, "world/model")

    assert isinstance(manifest, CsdRealizationManifest)
    assert result.blockers == ()
    assert manifest.backend == "gazebo"
    assert manifest.entry_file == "world.sdf"
    assert manifest.generated_files == ("manifest.json", "world.sdf", "assets/objects/mug.obj")
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest.to_json_dict()
    assert copied_mesh_path.is_file()
    assert root.attrib["version"] == "1.12"
    assert model.attrib["name"] == "mug"
    assert _required_element(model, "pose").text == "0.25 -0.1 0.82 0 0 0"
    assert _required_element(model, "static").text == "false"
    assert _required_element(model, "link/visual/geometry/mesh/uri").text == (
        "assets/objects/mug.obj"
    )
    assert _required_element(model, "link/collision/geometry/mesh/uri").text == (
        "assets/objects/mug.obj"
    )

    (asset_root / "objects" / "mug.obj").unlink()
    assert copied_mesh_path.is_file()


def test_compile_csd_to_gazebo_reports_missing_resource_adapter(tmp_path: Path) -> None:
    result = compile_csd_to_gazebo(
        csd={
            "csd_id": "csd_missing",
            "objects": [{"name": "mug", "asset_id": "object_mug"}],
        },
        asset_registry={"objects": [{"object_id": "object_mug", "variants": []}]},
        output_root=tmp_path,
        asset_root=tmp_path,
    )

    assert result.manifest is None
    assert result.blockers[0].to_json_dict() == {
        "blocker_id": "csd_missing_gazebo_object_mug_resource_missing",
        "csd_id": "csd_missing",
        "backend": "gazebo",
        "asset_id": "object_mug",
        "scope": "asset",
        "reason": "asset has no backend resource adapter for gazebo",
    }
