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


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
