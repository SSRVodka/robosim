from __future__ import annotations

from pathlib import Path
from shutil import copytree

import pytest
from pxr import Sdf, Usd

from robosim.core.openusd_csd import (
    compute_csd_digest,
    read_openusd_csd,
    validate_csd_stage,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures/csd/openusd/shared_tabletop"
CSD_PATH = FIXTURE_ROOT / "csd.usda"


def test_reader_extracts_typed_csd_handoff_and_backend_variant() -> None:
    csd = read_openusd_csd(CSD_PATH, backend="pybullet")

    assert csd.csd_id == "csd_shared_tabletop"
    assert csd.schema_version == "0.1"
    assert csd.task_instance_id == "task_shared_tabletop"
    assert csd.world_template_id == "world_tabletop"
    assert csd.backend == "pybullet"
    assert csd.task_objective == "keep the box away from the anchor"
    assert csd.randomization_seed == 17
    assert csd.randomization_values == {
        "lighting": "soft_left",
        "objectSpacingM": 0.5,
    }
    assert len(csd.digest) == 64
    assert {entity.entity_id for entity in csd.entities} == {
        "anchor",
        "dynamic_box",
        "environment",
        "robot",
        "table",
    }
    dynamic_box = next(entity for entity in csd.entities if entity.entity_id == "dynamic_box")
    assert dynamic_box.world_transform[12:15] == (0.0, 0.0, 0.35)

    relationship = csd.relationships[0]
    assert relationship.relationship_type == "avoid_contact"
    assert relationship.subject == Sdf.Path("/World/Objects/DynamicBox")
    assert relationship.object == Sdf.Path("/World/Objects/Anchor")
    assert relationship.min_distance_m == 0.25

    sensor = csd.sensors[0]
    assert sensor.sensor_type == "rgb"
    assert sensor.requirement == "required"
    assert sensor.min_resolution == (64, 64)
    assert sensor.source == Sdf.Path("/World/Camera")

    evaluator = csd.evaluators[0]
    assert evaluator.artifact_id == "predicate_keep_distance"
    assert evaluator.kind == "success_predicate"
    assert evaluator.path.name == "evaluator.json"
    assert evaluator.path.is_file()

    physics_scene = csd.stage.GetPrimAtPath("/World/PhysicsScene")
    assert "RobosimPyBulletSceneAPI" in physics_scene.GetMetadata("apiSchemas").GetAppliedItems()


def test_csd_digest_is_relocation_invariant_and_tracks_layer_content(
    tmp_path: Path,
) -> None:
    relocated = tmp_path / "relocated"
    copytree(FIXTURE_ROOT, relocated)

    original_digest = compute_csd_digest(CSD_PATH)
    relocated_digest = compute_csd_digest(relocated / "csd.usda")
    assert relocated_digest == original_digest

    task_layer = relocated / "layers" / "task.usda"
    task_layer.write_text(
        task_layer.read_text(encoding="utf-8").replace("soft_left", "soft_right"),
        encoding="utf-8",
    )

    assert compute_csd_digest(relocated / "csd.usda") != original_digest


def test_csd_digest_rejects_unresolved_dependencies(tmp_path: Path) -> None:
    relocated = tmp_path / "unresolved"
    copytree(FIXTURE_ROOT, relocated)
    (relocated / "evaluator.json").unlink()

    with pytest.raises(ValueError, match="unresolved CSD dependencies"):
        compute_csd_digest(relocated / "csd.usda")


def test_semantic_validation_reports_duplicate_entities_and_dangling_targets() -> None:
    csd = read_openusd_csd(CSD_PATH, backend="mujoco")
    stage = csd.stage
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetPrimAtPath("/World/Objects/Anchor").GetAttribute("robosim:entity:id").Set(
        "dynamic_box"
    )
    relationship = stage.GetPrimAtPath("/World/Relationships/BoxAvoidsAnchor")
    relationship.GetRelationship("robosim:relationship:object").SetTargets(
        [Sdf.Path("/World/Objects/Missing")]
    )

    issues = validate_csd_stage(stage)
    codes = {issue.code for issue in issues}

    assert "duplicate_entity_id" in codes
    assert "unresolved_relationship_target" in codes


def test_semantic_validation_rejects_nonconcrete_randomization() -> None:
    stage = Usd.Stage.Open(str(CSD_PATH))
    assert stage is not None
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetDefaultPrim().CreateAttribute(
        "robosim:randomization:value:choices",
        Sdf.ValueTypeNames.TokenArray,
        custom=True,
    ).Set(["left", "right"])

    issues = validate_csd_stage(stage)

    assert any(issue.code == "nonconcrete_randomization" for issue in issues)
