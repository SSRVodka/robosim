"""Core module for RoboSim framework."""

from robosim.core.capabilities import Capability
from robosim.core.csd import (
    CsdRealizationBlocker,
    CsdRealizationCacheKey,
    CsdRealizationManifest,
    asset_variant_hashes_for_csd,
    find_csd_realization_blockers,
    make_csd_realization_cache_key,
)

__all__ = [
    "CsdRealizationBlocker",
    "CsdRealizationCacheKey",
    "CsdRealizationManifest",
    "SimulatorBackend",
    "Capability",
    "asset_variant_hashes_for_csd",
    "find_csd_realization_blockers",
    "make_csd_realization_cache_key",
]


def __getattr__(name: str) -> object:
    if name == "SimulatorBackend":
        from robosim.core.backend import SimulatorBackend

        return SimulatorBackend
    raise AttributeError(name)
