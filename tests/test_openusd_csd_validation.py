from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from robosim.core.openusd_csd import csd_plugin_root, register_csd_plugins

FIXTURE_ROOT = Path(__file__).parent / "fixtures/csd/openusd/shared_tabletop"
CSD_PATH = FIXTURE_ROOT / "csd.usda"


def test_shared_csd_fixture_composes_complete_scenario() -> None:
    register_csd_plugins()
    assert CSD_PATH.is_file()
    stage = Usd.Stage.Open(str(CSD_PATH))

    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
    for path in (
        "/World/Environment/Surfaces/Table",
        "/World/Robot",
        "/World/Objects/DynamicBox",
        "/World/Objects/Anchor",
        "/World/Joints/BoxJoint",
        "/World/Camera",
        "/World/KeyLight",
        "/World/Task",
        "/World/Relationships/BoxAvoidsAnchor",
        "/World/Sensors/RgbCamera",
        "/World/Evaluators/KeepDistance",
    ):
        assert stage.GetPrimAtPath(path), path

    variants = stage.GetDefaultPrim().GetVariantSet("physicsBackend")
    assert variants.GetVariantSelection() == "mujoco"
    assert set(variants.GetVariantNames()) == {"gazebo", "mujoco", "pybullet"}

    variants.SetVariantSelection("pybullet")
    physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
    assert "RobosimPyBulletSceneAPI" in physics_scene.GetMetadata("apiSchemas").GetAppliedItems()
    assert physics_scene.GetAttribute("robosim:pybullet:numSolverIterations").Get() == 80

    variants.SetVariantSelection("gazebo")
    assert "RobosimGazeboSceneAPI" in physics_scene.GetMetadata("apiSchemas").GetAppliedItems()
    assert physics_scene.GetAttribute("robosim:gazebo:maxStepSize").Get() == pytest.approx(0.001)

    evaluator_path = (
        stage.GetPrimAtPath("/World/Evaluators/KeepDistance")
        .GetAttribute("robosim:evaluator:path")
        .Get()
    )
    assert Path(evaluator_path.resolvedPath).is_file()


def test_shared_csd_fixture_passes_strict_validation_for_every_backend() -> None:
    env = os.environ.copy()
    env["PXR_PLUGINPATH_NAME"] = str(csd_plugin_root())
    completed = subprocess.run(
        [
            "usdchecker",
            str(CSD_PATH),
            "--strict",
            "--variantSets",
            "physicsBackend",
        ],
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
