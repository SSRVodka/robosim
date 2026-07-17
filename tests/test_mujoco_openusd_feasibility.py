from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


FIXTURE_ROOT = Path(__file__).parent / "fixtures/csd/openusd/mujoco_feasibility"
CSD_PATH = FIXTURE_ROOT / "csd.usda"


def test_mujoco_openusd_fixture_is_composed_and_concrete() -> None:
    assert CSD_PATH.is_file()
    stage = Usd.Stage.Open(str(CSD_PATH))

    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
    assert stage.GetPrimAtPath("/World/Objects/DynamicBox")
    assert stage.GetPrimAtPath("/World/Objects/Anchor")
    assert stage.GetPrimAtPath("/World/Joints/BoxJoint")
    assert stage.GetPrimAtPath("/World/Camera")
    assert stage.GetPrimAtPath("/World/KeyLight")

    backend_variants = stage.GetDefaultPrim().GetVariantSet("physicsBackend")
    assert backend_variants.GetVariantSelection() == "mujoco"
    assert set(backend_variants.GetVariantNames()) == {"gazebo", "mujoco", "pybullet"}
    api_schemas = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene").GetPrim().GetMetadata(
        "apiSchemas"
    )
    assert "MjcSceneAPI" in api_schemas.GetAppliedItems()
