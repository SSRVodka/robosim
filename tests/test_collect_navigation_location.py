"""Tests for the semantic navigation location collection helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robosim.navigation.geometry import Pose2D

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "collect_navigation_location.py"
SPEC = importlib.util.spec_from_file_location("collect_navigation_location", SCRIPT_PATH)
assert SPEC is not None
collect_navigation_location = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_navigation_location)


def test_make_location_entry_rounds_pose_and_deduplicates_aliases() -> None:
    aliases = collect_navigation_location.parse_aliases(["客厅,living room", "客厅"])

    entry = collect_navigation_location.make_location_entry(
        Pose2D(8.12345, 1.98765, 0.33333),
        aliases=aliases,
        precision=2,
        nearest_waypoint_name="wp_living_room",
    )

    assert entry == {
        "aliases": ["客厅", "living room"],
        "target_pose": {"x": 8.12, "y": 1.99, "yaw": 0.33},
        "nearest_waypoint": "wp_living_room",
    }


def test_teleop_velocity_for_command_scales_known_commands() -> None:
    assert collect_navigation_location.teleop_velocity_for_command("w", 0.12, 0.8) == (
        0.12,
        0.0,
    )
    assert collect_navigation_location.teleop_velocity_for_command("a", 0.12, 0.8) == (
        0.0,
        0.8,
    )
    assert collect_navigation_location.teleop_velocity_for_command("x", 0.12, 0.8) == (
        0.0,
        0.0,
    )


def test_teleop_velocity_for_command_rejects_unknown_commands() -> None:
    assert collect_navigation_location.teleop_velocity_for_command("q", 0.12, 0.8) is None


def test_upsert_location_requires_overwrite_for_existing_key() -> None:
    config = {"locations": {"living_room": {}}}

    with pytest.raises(ValueError, match="already exists"):
        collect_navigation_location.upsert_location(
            config,
            "living_room",
            {"target_pose": {"x": 1.0, "y": 2.0, "yaw": 0.0}},
            overwrite=False,
        )


def test_save_config_writes_backup_for_existing_file(tmp_path: Path) -> None:
    locations_file = tmp_path / "locations.yaml"
    locations_file.write_text("scene: old\nlocations: {}\n", encoding="utf-8")

    collect_navigation_location.save_config(
        locations_file,
        {"scene": "new", "locations": {}},
        backup=True,
    )

    assert (tmp_path / "locations.yaml.bak").read_text(encoding="utf-8") == (
        "scene: old\nlocations: {}\n"
    )
    assert "scene: new" in locations_file.read_text(encoding="utf-8")
