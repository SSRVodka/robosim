"""CSD realization cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CsdRealizationCacheKey:
    """Deterministic cache key for one CSD/backend realization."""

    backend: str
    digest: str
    csd_hash: str
    asset_variant_hash: str
    realization_config_hash: str
    realization_version: str
    simulator_version: str | None


@dataclass(frozen=True, slots=True)
class CsdRealizationManifest:
    """Manifest for backend-native artifacts derived from one CSD."""

    manifest_id: str
    csd_id: str
    backend: str
    cache_key: str
    root_path: str
    entry_file: str
    generated_files: tuple[str, ...]
    preview_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.manifest_id:
            raise ValueError("manifest_id is required")
        if not self.csd_id:
            raise ValueError("csd_id is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.cache_key:
            raise ValueError("cache_key is required")
        if not self.root_path:
            raise ValueError("root_path is required")
        if not self.entry_file:
            raise ValueError("entry_file is required")
        object.__setattr__(
            self,
            "generated_files",
            tuple(str(path) for path in self.generated_files),
        )
        object.__setattr__(
            self,
            "preview_files",
            tuple(str(path) for path in self.preview_files),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "csd_id": self.csd_id,
            "backend": self.backend,
            "cache_key": self.cache_key,
            "root_path": self.root_path,
            "entry_file": self.entry_file,
            "generated_files": list(self.generated_files),
            "preview_files": list(self.preview_files),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CsdRealizationManifest":
        return cls(
            manifest_id=str(payload["manifest_id"]),
            csd_id=str(payload["csd_id"]),
            backend=str(payload["backend"]),
            cache_key=str(payload["cache_key"]),
            root_path=str(payload["root_path"]),
            entry_file=str(payload["entry_file"]),
            generated_files=tuple(
                str(path) for path in payload.get("generated_files", [])
            ),
            preview_files=tuple(str(path) for path in payload.get("preview_files", [])),
        )


@dataclass(frozen=True, slots=True)
class CsdRealizationBlocker:
    """Typed reason a CSD cannot be realized for one backend yet."""

    blocker_id: str
    csd_id: str
    backend: str
    asset_id: str
    scope: str
    reason: str

    def __post_init__(self) -> None:
        if not self.blocker_id:
            raise ValueError("blocker_id is required")
        if not self.csd_id:
            raise ValueError("csd_id is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.asset_id:
            raise ValueError("asset_id is required")
        if not self.scope:
            raise ValueError("scope is required")
        if not self.reason:
            raise ValueError("reason is required")

    def to_json_dict(self) -> dict[str, str]:
        return {
            "blocker_id": self.blocker_id,
            "csd_id": self.csd_id,
            "backend": self.backend,
            "asset_id": self.asset_id,
            "scope": self.scope,
            "reason": self.reason,
        }


def make_csd_realization_cache_key(
    *,
    csd: Mapping[str, Any],
    asset_variant_hashes: Mapping[str, str],
    backend: str,
    realization_config: Mapping[str, Any],
    realization_version: str,
    simulator_version: str | None,
) -> CsdRealizationCacheKey:
    """Build a stable cache key for derived backend-native scene artifacts."""
    if not backend:
        raise ValueError("backend is required")
    if not realization_version:
        raise ValueError("realization_version is required")
    csd_hash = _canonical_hash(csd)
    asset_variant_hash = _canonical_hash(asset_variant_hashes)
    realization_config_hash = _canonical_hash(realization_config)
    digest = _canonical_hash(
        {
            "backend": backend,
            "csd_hash": csd_hash,
            "asset_variant_hash": asset_variant_hash,
            "realization_config_hash": realization_config_hash,
            "realization_version": realization_version,
            "simulator_version": simulator_version,
        }
    )
    return CsdRealizationCacheKey(
        backend=backend,
        digest=digest,
        csd_hash=csd_hash,
        asset_variant_hash=asset_variant_hash,
        realization_config_hash=realization_config_hash,
        realization_version=realization_version,
        simulator_version=simulator_version,
    )


def find_csd_realization_blockers(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    backend: str,
) -> tuple[CsdRealizationBlocker, ...]:
    """Return blockers that prevent safe backend realization for a CSD."""
    if not backend:
        raise ValueError("backend is required")
    csd_id = _csd_id(csd)
    variants = _passed_variants_by_asset(asset_registry, backend)
    blockers: list[CsdRealizationBlocker] = []
    for asset_id in _csd_asset_ids(csd):
        if asset_id not in variants:
            blockers.append(
                CsdRealizationBlocker(
                    blocker_id=f"{csd_id}_{backend}_{asset_id}_variant_missing",
                    csd_id=csd_id,
                    backend=backend,
                    asset_id=asset_id,
                    scope="asset",
                    reason=f"asset has no passed backend variant for {backend}",
                )
            )
    return tuple(blockers)


def asset_variant_hashes_for_csd(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    backend: str,
) -> dict[str, str]:
    """Return backend variant hashes for all CSD assets, after compatibility checks."""
    blockers = find_csd_realization_blockers(
        csd=csd,
        asset_registry=asset_registry,
        backend=backend,
    )
    if blockers:
        raise ValueError("CSD has unresolved realization blockers")
    variants = _passed_variants_by_asset(asset_registry, backend)
    return {asset_id: str(variants[asset_id]["variant_hash"]) for asset_id in _csd_asset_ids(csd)}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csd_id(csd: Mapping[str, Any]) -> str:
    csd_id = str(csd.get("csd_id", ""))
    if not csd_id:
        raise ValueError("csd_id is required")
    return csd_id


def _csd_asset_ids(csd: Mapping[str, Any]) -> tuple[str, ...]:
    asset_ids: list[str] = []
    for obj in csd.get("objects", []):
        if isinstance(obj, Mapping) and obj.get("asset_id"):
            asset_ids.append(str(obj["asset_id"]))
    return tuple(dict.fromkeys(asset_ids))


def _passed_variants_by_asset(
    asset_registry: Mapping[str, Any],
    backend: str,
) -> dict[str, Mapping[str, Any]]:
    passed: dict[str, Mapping[str, Any]] = {}
    for obj in asset_registry.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        asset_id = str(obj.get("object_id") or obj.get("asset_id") or "")
        if not asset_id:
            continue
        variant = _passed_backend_variant(obj.get("variants", ()), backend)
        if variant is not None:
            passed[asset_id] = variant
    for asset in asset_registry.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        asset_id = str(asset.get("asset_id", ""))
        if not asset_id or asset_id in passed:
            continue
        variant = _passed_backend_variant(asset.get("variants", ()), backend)
        if variant is not None:
            passed[asset_id] = variant
    return passed


def _passed_backend_variant(
    variants: object,
    backend: str,
) -> Mapping[str, Any] | None:
    if not isinstance(variants, (list, tuple)):
        return None
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        if str(variant.get("engine", "")) != backend:
            continue
        if str(variant.get("validation_state", "")).lower() != "passed":
            continue
        if not variant.get("variant_hash"):
            continue
        return variant
    return None
