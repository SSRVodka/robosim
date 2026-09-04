"""Dispatch self-contained OpenUSD CSD packages to backend compilers."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robosim.core.csd import CsdRealizationBlocker, CsdRealizationManifest
from robosim.core.gazebo_openusd_package import (
    compile_openusd_scene_package as compile_openusd_gazebo_scene_package,
)
from robosim.core.mujoco_openusd_package import PackageError
from robosim.core.mujoco_openusd_package import (
    compile_openusd_scene_package as compile_openusd_mujoco_scene_package,
)
from robosim.core.pybullet_openusd_package import (
    compile_openusd_scene_package as compile_openusd_pybullet_scene_package,
)

MUJOCO_BACKEND = "mujoco"
GAZEBO_BACKEND = "gazebo"
PYBULLET_BACKEND = "pybullet"
DEFAULT_REALIZATION_VERSION = "csd-compiler-0.10"

_Compiler = Callable[..., CsdRealizationManifest]


@dataclass(frozen=True, slots=True)
class CsdCompilationResult:
    """Result of compiling one fixed CSD into backend-native artifacts."""

    manifest: CsdRealizationManifest | None
    blockers: tuple[CsdRealizationBlocker, ...] = ()


def _compile(
    *,
    backend: str,
    compiler: _Compiler,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None,
    realization_version: str,
    simulator_version: str | None,
) -> CsdCompilationResult:
    expected_root = csd_path.resolve().parent / "engine_manifests"
    if output_root.resolve() != expected_root:
        csd_id = csd_path.resolve().parent.name
        return CsdCompilationResult(
            manifest=None,
            blockers=(
                CsdRealizationBlocker(
                    blocker_id=f"{csd_id}_{backend}_invalid_output_root",
                    csd_id=csd_id,
                    backend=backend,
                    asset_id="openusd_package",
                    scope="csd",
                    reason=(
                        "output_root must be the scene package engine_manifests directory: "
                        f"{expected_root}"
                    ),
                ),
            ),
        )
    try:
        return CsdCompilationResult(
            manifest=compiler(
                csd_path=csd_path,
                output_root=output_root,
                realization_config=realization_config,
                realization_version=realization_version,
                simulator_version=simulator_version,
            )
        )
    except (ImportError, PackageError, OSError, ValueError) as error:
        csd_id = Path(csd_path).parent.name
        return CsdCompilationResult(
            manifest=None,
            blockers=(
                CsdRealizationBlocker(
                    blocker_id=f"{csd_id}_{backend}_openusd_package_compile_blocked",
                    csd_id=csd_id,
                    backend=backend,
                    asset_id="openusd_package",
                    scope="csd",
                    reason=str(error),
                ),
            ),
        )


def compile_csd_to_mujoco(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile a self-contained OpenUSD CSD package into MJCF."""
    return _compile(
        backend=MUJOCO_BACKEND,
        compiler=compile_openusd_mujoco_scene_package,
        csd_path=csd_path,
        output_root=output_root,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version or _mujoco_simulator_version(),
    )


def compile_csd_to_pybullet(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile a self-contained OpenUSD CSD package for PyBullet."""
    return _compile(
        backend=PYBULLET_BACKEND,
        compiler=compile_openusd_pybullet_scene_package,
        csd_path=csd_path,
        output_root=output_root,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version or _pybullet_simulator_version(),
    )


def compile_csd_to_gazebo(
    *,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile a self-contained OpenUSD CSD package for Gazebo Classic."""
    return _compile(
        backend=GAZEBO_BACKEND,
        compiler=compile_openusd_gazebo_scene_package,
        csd_path=csd_path,
        output_root=output_root,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version or _gazebo_simulator_version(),
    )


def compile_csd(
    *,
    backend: str,
    csd_path: Path,
    output_root: Path,
    realization_config: Mapping[str, Any] | None = None,
    realization_version: str = DEFAULT_REALIZATION_VERSION,
    simulator_version: str | None = None,
) -> CsdCompilationResult:
    """Compile one self-contained OpenUSD CSD package for a backend target."""
    compilers = {
        MUJOCO_BACKEND: compile_csd_to_mujoco,
        PYBULLET_BACKEND: compile_csd_to_pybullet,
        GAZEBO_BACKEND: compile_csd_to_gazebo,
    }
    backend_key = backend.strip().lower()
    if backend_key not in compilers:
        raise ValueError(f"unsupported CSD compiler backend: {backend}")
    return compilers[backend_key](
        csd_path=csd_path,
        output_root=output_root,
        realization_config=realization_config,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )


def _mujoco_simulator_version() -> str | None:
    try:
        import mujoco
    except Exception:
        return None
    return str(getattr(mujoco, "__version__", "")) or None


def _pybullet_simulator_version() -> str | None:
    try:
        import pybullet as pybullet
    except Exception:
        return None
    return str(pybullet.getAPIVersion())


def _gazebo_simulator_version() -> str | None:
    try:
        result = subprocess.run(
            ["gazebo", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if "version" in line.lower():
            return line.rsplit("version", 1)[-1].strip() or None
    return None
