"""Tests for the thin OpenUSD backend compiler dispatcher."""

from pathlib import Path

import pytest

from robosim.core import csd_compiler
from robosim.core.csd_compiler import CsdCompilationResult, compile_csd
from robosim.core.mujoco_openusd_package import PackageError


@pytest.mark.parametrize("backend", ("mujoco", "pybullet", "gazebo"))
def test_compile_csd_dispatches_to_backend(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = CsdCompilationResult(manifest=None)
    calls: list[dict[str, object]] = []

    def compiler(**kwargs: object) -> CsdCompilationResult:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(csd_compiler, f"compile_csd_to_{backend}", compiler)
    result = compile_csd(
        backend=backend.upper(),
        csd_path=tmp_path / "scene.usda",
        output_root=tmp_path / "engine_manifests",
        simulator_version="test",
    )

    assert result is expected
    assert calls[0]["simulator_version"] == "test"


def test_compile_csd_reports_package_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compiler(**_: object) -> None:
        raise PackageError("invalid package")

    monkeypatch.setattr(csd_compiler, "compile_openusd_gazebo_scene_package", compiler)
    result = csd_compiler.compile_csd_to_gazebo(
        csd_path=tmp_path / "scene.usda",
        output_root=tmp_path,
        simulator_version="test",
    )

    assert result.manifest is None
    assert result.blockers[0].backend == "gazebo"
    assert result.blockers[0].reason == "invalid package"


def test_compile_csd_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported CSD compiler backend"):
        compile_csd(
            backend="unknown",
            csd_path=tmp_path / "scene.usda",
            output_root=tmp_path,
        )
