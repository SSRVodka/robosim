from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

from robosim.core.openusd_csd import csd_plugin_root, register_csd_plugins

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "csd" / "openusd"


def test_codeless_csd_schemas_register_in_fresh_process() -> None:
    probe = """
import json
from pxr import Usd
from robosim.core.openusd_csd import register_csd_plugins

register_csd_plugins()
registry = Usd.SchemaRegistry()
result = {
    "apis": {
        name: bool(registry.FindAppliedAPIPrimDefinition(name))
        for name in (
            "RobosimCsdRootAPI",
            "RobosimEntityAPI",
            "RobosimTaskAPI",
            "RobosimRandomizationAPI",
            "RobosimPyBulletSceneAPI",
            "RobosimGazeboSceneAPI",
        )
    },
    "types": {
        name: bool(registry.FindConcretePrimDefinition(name))
        for name in (
            "RobosimRelationship",
            "RobosimSensorRequirement",
            "RobosimEvaluatorRef",
        )
    },
}
print(json.dumps(result, sort_keys=True))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert all(result["apis"].values())
    assert all(result["types"].values())


def test_codeless_csd_schema_authors_a_strictly_valid_layer(tmp_path: Path) -> None:
    register_csd_plugins()
    stage_path = tmp_path / "minimal_csd.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world.AddAppliedSchema("RobosimCsdRootAPI")
    for name, value in {
        "robosim:csd:id": "csd_minimal",
        "robosim:csd:schemaVersion": "0.1",
        "robosim:csd:taskInstanceId": "task_minimal",
        "robosim:csd:worldTemplateId": "world_empty",
    }.items():
        world.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)

    subject = UsdGeom.Xform.Define(stage, "/World/Objects/Subject").GetPrim()
    target = UsdGeom.Xform.Define(stage, "/World/Objects/Target").GetPrim()
    relationship = stage.DefinePrim("/World/Relationships/SubjectNearTarget", "RobosimRelationship")
    relationship.CreateAttribute("robosim:relationship:type", Sdf.ValueTypeNames.Token).Set("near")
    relationship.CreateRelationship("robosim:relationship:subject").SetTargets([subject.GetPath()])
    relationship.CreateRelationship("robosim:relationship:object").SetTargets([target.GetPath()])
    relationship.CreateAttribute(
        "robosim:relationship:minDistanceM", Sdf.ValueTypeNames.Double
    ).Set(0.1)
    stage.GetRootLayer().Save()

    assert "RobosimCsdRootAPI" in world.GetMetadata("apiSchemas").GetAppliedItems()
    assert relationship.GetTypeName() == "RobosimRelationship"
    assert relationship.GetAttribute("robosim:relationship:type").Get() == "near"

    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(csd_plugin_root())
    completed = subprocess.run(
        ["usdchecker", str(stage_path), "--strict"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_committed_csd_fixtures_pass_official_openusd_validation() -> None:
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(csd_plugin_root())
    fixture_paths = (
        FIXTURE_ROOT / "shared_tabletop" / "csd.usda",
        *sorted((FIXTURE_ROOT / "semantic").glob("*/csd.usda")),
    )

    for fixture_path in fixture_paths:
        completed = subprocess.run(
            ["usdchecker", str(fixture_path)],
            capture_output=True,
            env=env,
            text=True,
        )
        assert completed.returncode == 0, f"{fixture_path}\n{completed.stdout}{completed.stderr}"
