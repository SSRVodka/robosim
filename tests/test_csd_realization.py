"""Tests for CSD realization cache contracts."""

from robosim.core.csd import make_csd_realization_cache_key


def test_csd_realization_cache_key_helper_is_core_api() -> None:
    from robosim.core import make_csd_realization_cache_key as exported

    assert exported is make_csd_realization_cache_key


def test_csd_realization_cache_key_is_stable_for_mapping_order() -> None:
    left = make_csd_realization_cache_key(
        csd={"csd_id": "csd_0001", "objects": [{"asset_id": "asset_mug"}]},
        asset_variant_hashes={"asset_mug": "hash_mug", "asset_table": "hash_table"},
        backend="mujoco",
        realization_config={"quality": "preview", "headless": True},
        realization_version="0.1",
        simulator_version="3.2.0",
    )
    right = make_csd_realization_cache_key(
        csd={"objects": [{"asset_id": "asset_mug"}], "csd_id": "csd_0001"},
        asset_variant_hashes={"asset_table": "hash_table", "asset_mug": "hash_mug"},
        backend="mujoco",
        realization_config={"headless": True, "quality": "preview"},
        realization_version="0.1",
        simulator_version="3.2.0",
    )

    assert right == left
    assert left.digest
    assert left.backend == "mujoco"


def test_csd_realization_cache_key_changes_by_backend() -> None:
    mujoco = make_csd_realization_cache_key(
        csd={"csd_id": "csd_0001"},
        asset_variant_hashes={},
        backend="mujoco",
        realization_config={},
        realization_version="0.1",
        simulator_version=None,
    )
    gazebo = make_csd_realization_cache_key(
        csd={"csd_id": "csd_0001"},
        asset_variant_hashes={},
        backend="gazebo",
        realization_config={},
        realization_version="0.1",
        simulator_version=None,
    )

    assert gazebo.digest != mujoco.digest
