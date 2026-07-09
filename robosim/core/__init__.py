"""Core module for RoboSim framework."""

from robosim.core.capabilities import Capability
from robosim.core.csd import (
    ConcreteScenarioDefinition,
    CsdRealizationBlocker,
    CsdRealizationCacheKey,
    CsdRealizationManifest,
    CsdRelationshipType,
    asset_variant_hashes_for_csd,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)
from robosim.core.csd_compiler import (
    CsdCompilationResult,
    compile_csd,
    compile_csd_to_gazebo,
    compile_csd_to_mujoco,
)

__all__ = [
    "CsdCompilationResult",
    "ConcreteScenarioDefinition",
    "CsdRealizationBlocker",
    "CsdRealizationCacheKey",
    "CsdRealizationManifest",
    "CsdRelationshipType",
    "SimulatorBackend",
    "Capability",
    "asset_variant_hashes_for_csd",
    "compile_csd",
    "compile_csd_to_gazebo",
    "compile_csd_to_mujoco",
    "find_csd_realization_blockers",
    "make_csd_realization_cache_key",
]


def __getattr__(name: str) -> object:
    if name == "SimulatorBackend":
        from robosim.core.backend import SimulatorBackend

        return SimulatorBackend
    raise AttributeError(name)
