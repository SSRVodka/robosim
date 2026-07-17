"""Core module for RoboSim framework."""

from robosim.core.capabilities import Capability
from robosim.core.csd import (
    BackendResourceAdapter,
    BackendResourceMaterial,
    ConcreteScenarioDefinition,
    CsdObjectContact,
    CsdObjectInertial,
    CsdObjectInitialState,
    CsdRealizationBlocker,
    CsdRealizationCacheKey,
    CsdRealizationManifest,
    CsdRealizationValidationRecord,
    CsdRelationshipType,
    backend_resource_adapters_by_asset,
    make_csd_realization_cache_key,
)
from robosim.core.csd_compiler import (
    CsdCompilationResult,
    compile_csd,
    compile_csd_to_gazebo,
    compile_csd_to_mujoco,
    compile_csd_to_pybullet,
)
from robosim.core.openusd_csd import (
    CsdStageValidationIssue,
    OpenUsdCsd,
    compute_csd_digest,
    read_openusd_csd,
    register_csd_plugins,
    validate_csd_stage,
)

__all__ = [
    "CsdCompilationResult",
    "BackendResourceAdapter",
    "BackendResourceMaterial",
    "ConcreteScenarioDefinition",
    "CsdObjectContact",
    "CsdObjectInitialState",
    "CsdObjectInertial",
    "CsdRealizationBlocker",
    "CsdRealizationCacheKey",
    "CsdRealizationManifest",
    "CsdRealizationValidationRecord",
    "CsdRelationshipType",
    "CsdStageValidationIssue",
    "OpenUsdCsd",
    "SimulatorBackend",
    "Capability",
    "backend_resource_adapters_by_asset",
    "compile_csd",
    "compile_csd_to_gazebo",
    "compile_csd_to_mujoco",
    "compile_csd_to_pybullet",
    "compute_csd_digest",
    "make_csd_realization_cache_key",
    "read_openusd_csd",
    "register_csd_plugins",
    "validate_csd_stage",
]


def __getattr__(name: str) -> object:
    if name == "SimulatorBackend":
        from robosim.core.backend import SimulatorBackend

        return SimulatorBackend
    raise AttributeError(name)
