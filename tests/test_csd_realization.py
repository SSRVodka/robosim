"""Tests for CSD realization cache contracts."""

from robosim.core.csd import (
    CsdRealizationManifest,
    CsdRealizationValidationRecord,
    make_csd_realization_cache_key,
)


def test_csd_realization_cache_key_helper_is_core_api() -> None:
    from robosim.core import make_csd_realization_cache_key as exported

    assert exported is make_csd_realization_cache_key


def test_csd_realization_cache_key_changes_by_backend() -> None:
    mujoco = make_csd_realization_cache_key(
        csd_hash="a" * 64,
        asset_variant_hashes={},
        backend="mujoco",
        realization_config={},
        realization_version="0.1",
        simulator_version=None,
    )
    gazebo = make_csd_realization_cache_key(
        csd_hash="a" * 64,
        asset_variant_hashes={},
        backend="gazebo",
        realization_config={},
        realization_version="0.1",
        simulator_version=None,
    )

    assert gazebo.digest != mujoco.digest


def test_csd_realization_cache_key_accepts_composed_stage_digest() -> None:
    key = make_csd_realization_cache_key(
        csd_hash="a" * 64,
        asset_variant_hashes={"object_box": "resource-hash"},
        backend="mujoco",
        realization_config={},
        realization_version="0.4",
        simulator_version="3.9.0",
    )

    assert key.csd_hash == "a" * 64


def test_csd_realization_manifest_round_trips_backend_artifacts() -> None:
    manifest = CsdRealizationManifest(
        manifest_id="manifest_mujoco_csd_0001",
        csd_id="csd_0001",
        backend="mujoco",
        cache_key="abc123",
        root_path="engine_manifests/mujoco/csd_0001",
        entry_file="scene.xml",
        generated_files=("scene.xml", "assets/mug.obj"),
        preview_files=("render_previews/mujoco/csd_0001/front.png",),
    )

    restored = CsdRealizationManifest.from_json_dict(manifest.to_json_dict())

    assert restored == manifest


def test_csd_realization_validation_record_round_trips() -> None:
    record = CsdRealizationValidationRecord(
        validation_id="validation_mujoco_csd_0001",
        csd_id="csd_0001",
        backend="mujoco",
        manifest_id="manifest_mujoco_csd_0001",
        cache_key="abc123",
        status="passed",
        evidence_files=(
            "diagnostics/load_check.json",
            "diagnostics/physics_check.json",
        ),
        preview_files=("diagnostics/semantic_preview.ppm",),
    )

    restored = CsdRealizationValidationRecord.from_json_dict(record.to_json_dict())

    assert restored == record
