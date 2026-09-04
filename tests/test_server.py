"""Tests for server backend construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from robosim import server


def test_importing_mujoco_backend_does_not_load_gazebo_or_rclpy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import robosim.backends.mujoco; "
                "assert 'robosim.backends.gazebo.backend' not in sys.modules; "
                "assert 'rclpy' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_create_mujoco_backend_from_csd_manifest(monkeypatch) -> None:
    calls: list[tuple[Path, bool]] = []

    class FakeMuJoCoBackend:
        @classmethod
        def from_csd_realization_manifest_file(
            cls,
            manifest_path: Path,
            *,
            headless: bool = True,
        ) -> "FakeMuJoCoBackend":
            calls.append((manifest_path, headless))
            return cls()

    monkeypatch.setattr(server, "_load_backend_class", lambda _: FakeMuJoCoBackend)

    backend = server.create_backend(
        backend_type="mujoco",
        robot_name="ignored",
        scene=None,
        csd_manifest="/tmp/engine_manifests/mujoco/csd_0001/manifest.json",
        headless=False,
    )

    assert isinstance(backend, FakeMuJoCoBackend)
    assert calls == [
        (Path("/tmp/engine_manifests/mujoco/csd_0001/manifest.json"), False)
    ]


def test_create_mujoco_backend_from_scene_path(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeMuJoCoBackend:
        def __init__(self, *, scene_path: str, headless: bool = True) -> None:
            calls.append((scene_path, headless))

    monkeypatch.setattr(server, "_load_backend_class", lambda _: FakeMuJoCoBackend)

    backend = server.create_backend(
        backend_type="mujoco",
        robot_name="ignored",
        scene="/tmp/scene.xml",
        csd_manifest=None,
        headless=False,
    )

    assert isinstance(backend, FakeMuJoCoBackend)
    assert calls == [("/tmp/scene.xml", False)]


def test_create_pybullet_backend_from_csd_manifest(monkeypatch) -> None:
    calls: list[tuple[Path, bool]] = []

    class FakePyBulletBackend:
        @classmethod
        def from_csd_realization_manifest_file(
            cls,
            manifest_path: Path,
            *,
            headless: bool = True,
        ) -> "FakePyBulletBackend":
            calls.append((manifest_path, headless))
            return cls()

    monkeypatch.setattr(server, "_load_backend_class", lambda _: FakePyBulletBackend)

    backend = server.create_backend(
        backend_type="pybullet",
        robot_name="ignored",
        scene=None,
        csd_manifest="/tmp/engine_manifests/pybullet/csd_0001/manifest.json",
        headless=False,
    )

    assert isinstance(backend, FakePyBulletBackend)
    assert calls == [
        (Path("/tmp/engine_manifests/pybullet/csd_0001/manifest.json"), False)
    ]


def test_create_pybullet_backend_from_default_scene(monkeypatch) -> None:
    calls: list[tuple[str | None, bool]] = []

    class FakePyBulletBackend:
        def __init__(self, *, scene_path: str | None = None, headless: bool = True) -> None:
            calls.append((scene_path, headless))

    monkeypatch.setattr(server, "_load_backend_class", lambda _: FakePyBulletBackend)

    backend = server.create_backend(
        backend_type="pybullet",
        robot_name="ignored",
        scene=None,
        csd_manifest=None,
        headless=True,
    )

    assert isinstance(backend, FakePyBulletBackend)
    assert calls == [(None, True)]


def test_no_headless_selects_backend_viewer_mode() -> None:
    args = server.parse_args(("--backend", "mujoco", "--no-headless"))

    assert args.backend == "mujoco"
    assert args.headless is False


def test_no_headless_is_preserved_for_gazebo_server() -> None:
    args = server.parse_args(("--backend", "gazebo", "--no-headless"))

    assert args.headless is False
