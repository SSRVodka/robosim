"""OpenUSD CSD schema registration and stage helpers."""

from __future__ import annotations

from pathlib import Path

from pxr import Plug

_PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "usd_plugins" / "robosimCsd"


def csd_plugin_root() -> Path:
    """Return the packaged codeless CSD schema resource directory."""
    return _PLUGIN_ROOT


def register_csd_plugins() -> None:
    """Register the packaged codeless CSD schemas with OpenUSD."""
    Plug.Registry().RegisterPlugins(str(_PLUGIN_ROOT / "plugInfo.json"))

