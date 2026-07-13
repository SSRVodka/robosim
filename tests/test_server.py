"""Tests for server backend construction."""

from __future__ import annotations

from pathlib import Path

from robosim import server


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

    monkeypatch.setattr(server, "MuJoCoBackend", FakeMuJoCoBackend)

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

    monkeypatch.setattr(server, "MuJoCoBackend", FakeMuJoCoBackend)

    backend = server.create_backend(
        backend_type="mujoco",
        robot_name="ignored",
        scene="/tmp/scene.xml",
        csd_manifest=None,
        headless=True,
    )

    assert isinstance(backend, FakeMuJoCoBackend)
    assert calls == [("/tmp/scene.xml", True)]


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

    monkeypatch.setattr(server, "PyBulletBackend", FakePyBulletBackend)

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

    monkeypatch.setattr(server, "PyBulletBackend", FakePyBulletBackend)

    backend = server.create_backend(
        backend_type="pybullet",
        robot_name="ignored",
        scene=None,
        csd_manifest=None,
        headless=True,
    )

    assert isinstance(backend, FakePyBulletBackend)
    assert calls == [(None, True)]
