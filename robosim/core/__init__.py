"""Core module for RoboSim framework."""

from robosim.core.capabilities import Capability
from robosim.core.csd import CsdRealizationCacheKey, make_csd_realization_cache_key

__all__ = [
    "CsdRealizationCacheKey",
    "SimulatorBackend",
    "Capability",
    "make_csd_realization_cache_key",
]


def __getattr__(name: str) -> object:
    if name == "SimulatorBackend":
        from robosim.core.backend import SimulatorBackend

        return SimulatorBackend
    raise AttributeError(name)
