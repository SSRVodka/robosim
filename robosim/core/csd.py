"""CSD realization cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

DEFAULT_MUJOCO_OBJECT_FRICTION = (0.7, 0.005, 0.0001)


class CsdRelationshipType(StrEnum):
    """Machine-readable CSD relationship types supported by the compiler."""

    ON_TOP_OF = "on_top_of"
    INSIDE = "inside"
    NEAR = "near"
    AVOID_CONTACT = "avoid_contact"
    ALIGNED_WITH = "aligned_with"
    ATTACHED_TO = "attached_to"


@dataclass(frozen=True, slots=True)
class CsdVector3:
    """Three-dimensional vector in CSD units."""

    x: float
    y: float
    z: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, field: str) -> "CsdVector3":
        return cls(
            x=_required_float(payload, "x", field=field),
            y=_required_float(payload, "y", field=field),
            z=_required_float(payload, "z", field=field),
        )


@dataclass(frozen=True, slots=True)
class CsdQuaternion:
    """Quaternion stored in MuJoCo-compatible wxyz order."""

    w: float
    x: float
    y: float
    z: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, field: str) -> "CsdQuaternion":
        return cls(
            w=_required_float(payload, "w", field=field),
            x=_required_float(payload, "x", field=field),
            y=_required_float(payload, "y", field=field),
            z=_required_float(payload, "z", field=field),
        )


@dataclass(frozen=True, slots=True)
class CsdPose:
    """Pose in the CSD root frame."""

    position: CsdVector3
    orientation: CsdQuaternion

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, field: str) -> "CsdPose":
        return cls(
            position=CsdVector3.from_mapping(
                _required_mapping(payload, "position", field=field),
                field=f"{field}.position",
            ),
            orientation=CsdQuaternion.from_mapping(
                _required_mapping(payload, "orientation", field=field),
                field=f"{field}.orientation",
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdSurface:
    """Backend-neutral static environment surface."""

    surface_id: str
    surface_type: str
    pose: CsdPose
    size: CsdVector3
    rgba: tuple[float, float, float, float]
    friction: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdSurface":
        return cls(
            surface_id=_required_str(payload, "surface_id", field="surface"),
            surface_type=_required_str(payload, "type", field="surface"),
            pose=CsdPose.from_mapping(
                _required_mapping(payload, "pose", field="surface"),
                field="surface.pose",
            ),
            size=CsdVector3.from_mapping(
                _required_mapping(payload, "size", field="surface"),
                field="surface.size",
            ),
            rgba=_number_tuple(
                payload.get("rgba", (0.42, 0.36, 0.28, 1.0)),
                length=4,
                field="surface.rgba",
            ),
            friction=_number_tuple(
                payload.get("friction", (1.2, 0.2, 0.2)),
                length=3,
                field="surface.friction",
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdCamera:
    """CSD camera requested by a scenario."""

    camera_id: str
    position: CsdVector3
    xyaxes: tuple[float, float, float, float, float, float] | None
    mode: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdCamera":
        pose = _required_mapping(payload, "pose", field="camera")
        position = CsdVector3.from_mapping(
            _required_mapping(pose, "position", field="camera.pose"),
            field="camera.pose.position",
        )
        xyaxes = None
        if pose.get("xyaxes") is not None:
            xyaxes = _number_tuple(pose["xyaxes"], length=6, field="camera.pose.xyaxes")
        return cls(
            camera_id=_required_str(payload, "camera_id", field="camera"),
            position=position,
            xyaxes=xyaxes,
            mode=str(payload.get("mode", "fixed")),
        )


@dataclass(frozen=True, slots=True)
class CsdLight:
    """CSD light source."""

    light_id: str
    position: CsdVector3
    direction: CsdVector3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdLight":
        return cls(
            light_id=_required_str(payload, "light_id", field="light"),
            position=_vector_from_sequence(payload.get("position"), field="light.position"),
            direction=_vector_from_sequence(payload.get("direction"), field="light.direction"),
        )


@dataclass(frozen=True, slots=True)
class CsdEnvironment:
    """Global background/environment portion of a CSD scenario."""

    environment_id: str
    environment_type: str
    gravity: CsdVector3
    surfaces: tuple[CsdSurface, ...]
    cameras: tuple[CsdCamera, ...]
    lighting: tuple[CsdLight, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdEnvironment":
        return cls(
            environment_id=_required_str(payload, "environment_id", field="environment"),
            environment_type=_required_str(payload, "type", field="environment"),
            gravity=_vector_from_sequence(
                payload.get("gravity", (0.0, 0.0, -9.81)),
                field="environment.gravity",
            ),
            surfaces=tuple(
                CsdSurface.from_mapping(surface)
                for surface in _mapping_list(
                    payload.get("surfaces", ()),
                    field="environment.surfaces",
                )
            ),
            cameras=tuple(
                CsdCamera.from_mapping(camera)
                for camera in _mapping_list(payload.get("cameras", ()), field="environment.cameras")
            ),
            lighting=tuple(
                CsdLight.from_mapping(light)
                for light in _mapping_list(
                    payload.get("lighting", ()),
                    field="environment.lighting",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdRobot:
    """Robot instance requested by a CSD scenario."""

    asset_id: str
    pose: CsdPose

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdRobot":
        pose = payload.get("pose")
        return cls(
            asset_id=_required_str(payload, "asset_id", field="robot"),
            pose=(
                CsdPose.from_mapping(pose, field="robot.pose")
                if isinstance(pose, Mapping)
                else _identity_pose()
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdObjectContact:
    """Optional backend-neutral contact parameters for an object geom."""

    margin_m: float | None
    gap_m: float | None
    solref: tuple[float, float] | None
    solimp: tuple[float, float, float, float, float] | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdObjectContact":
        margin = payload.get("margin_m")
        gap = payload.get("gap_m")
        solref = None
        if payload.get("solref") is not None:
            solref = _number_tuple(payload["solref"], length=2, field="contact.solref")
        solimp = None
        if payload.get("solimp") is not None:
            solimp = _number_tuple(payload["solimp"], length=5, field="contact.solimp")
        return cls(
            margin_m=float(margin) if margin is not None else None,
            gap_m=float(gap) if gap is not None else None,
            solref=solref,
            solimp=solimp,
        )


@dataclass(frozen=True, slots=True)
class CsdObjectInertial:
    """Optional explicit object inertia in the object body frame."""

    center_of_mass: CsdVector3
    diagonal_inertia_kg_m2: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdObjectInertial":
        return cls(
            center_of_mass=CsdVector3.from_mapping(
                _required_mapping(payload, "center_of_mass", field="inertial"),
                field="inertial.center_of_mass",
            ),
            diagonal_inertia_kg_m2=_number_tuple(
                payload.get("diagonal_inertia_kg_m2"),
                length=3,
                field="inertial.diagonal_inertia_kg_m2",
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdObjectInitialState:
    """Backend-neutral object physical state requested by CSD."""

    mass_kg: float
    friction: tuple[float, float, float]
    contact: CsdObjectContact | None
    inertial: CsdObjectInertial | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdObjectInitialState":
        contact_payload = payload.get("contact")
        inertial_payload = payload.get("inertial")
        return cls(
            mass_kg=float(payload.get("mass_kg", 0.1)),
            friction=_friction_tuple(payload.get("friction", DEFAULT_MUJOCO_OBJECT_FRICTION)),
            contact=(
                CsdObjectContact.from_mapping(contact_payload)
                if isinstance(contact_payload, Mapping)
                else None
            ),
            inertial=(
                CsdObjectInertial.from_mapping(inertial_payload)
                if isinstance(inertial_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class CsdObject:
    """Concrete object instance in a CSD scenario."""

    name: str
    asset_id: str
    role: str
    pose: CsdPose
    static: bool
    initial_state: CsdObjectInitialState
    rgba: tuple[float, float, float, float] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdObject":
        return cls(
            name=_required_str(payload, "name", field="object"),
            asset_id=_required_str(payload, "asset_id", field="object"),
            role=str(payload.get("role", "")),
            pose=CsdPose.from_mapping(
                _required_mapping(payload, "pose", field="object"),
                field="object.pose",
            ),
            static=bool(payload.get("static", False)),
            initial_state=CsdObjectInitialState.from_mapping(
                _optional_mapping(payload, "initial_state")
            ),
            rgba=None,
        )


@dataclass(frozen=True, slots=True)
class CsdRelationship:
    """Compiler-readable CSD relationship."""

    relation_id: str
    type: CsdRelationshipType
    subject: str
    object: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CsdRelationship":
        relation_type = _required_str(payload, "type", field="relationship")
        try:
            typed_relation = CsdRelationshipType(relation_type)
        except ValueError as exc:
            raise ValueError(f"unsupported relationship type: {relation_type}") from exc
        return cls(
            relation_id=_required_str(payload, "relation_id", field="relationship"),
            type=typed_relation,
            subject=_required_str(payload, "subject", field="relationship"),
            object=_required_str(payload, "object", field="relationship"),
            parameters=_optional_mapping(payload, "parameters"),
        )


@dataclass(frozen=True, slots=True)
class ConcreteScenarioDefinition:
    """Typed CSD scenario consumed by backend compilers."""

    csd_id: str
    schema_version: str
    frame: str
    units: str
    environment: CsdEnvironment
    robot: CsdRobot | None
    objects: tuple[CsdObject, ...]
    relationships: tuple[CsdRelationship, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConcreteScenarioDefinition":
        scenario = _required_mapping(payload, "scenario", field="CSD")
        robot_payload = scenario.get("robot")
        return cls(
            csd_id=_required_str(payload, "csd_id", field="CSD"),
            schema_version=str(payload.get("schema_version", "")),
            frame=str(scenario.get("frame", "world")),
            units=str(scenario.get("units", "m")),
            environment=CsdEnvironment.from_mapping(
                _required_mapping(scenario, "environment", field="scenario")
            ),
            robot=(
                CsdRobot.from_mapping(robot_payload) if isinstance(robot_payload, Mapping) else None
            ),
            objects=tuple(
                CsdObject.from_mapping(obj)
                for obj in _mapping_list(scenario.get("objects", ()), field="scenario.objects")
            ),
            relationships=tuple(
                CsdRelationship.from_mapping(rel)
                for rel in _mapping_list(
                    scenario.get("relationships", ()),
                    field="scenario.relationships",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class BackendResourceMaterial:
    """Material/media adapter attached to a backend resource."""

    name: str | None
    rgba: tuple[float, float, float, float] | None
    texture_path: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BackendResourceMaterial":
        rgba = None
        if payload.get("rgba") is not None:
            rgba = _number_tuple(payload["rgba"], length=4, field="material.rgba")
        texture_path = payload.get("texture_path")
        return cls(
            name=str(payload["name"]) if payload.get("name") else None,
            rgba=rgba,
            texture_path=str(texture_path) if texture_path else None,
        )


@dataclass(frozen=True, slots=True)
class BackendResourceAdapter:
    """Concrete backend resource adapter for one accepted asset."""

    asset_id: str
    backend: str
    resource_id: str | None
    mesh_path: str
    resource_hash: str
    mesh_scale: float | tuple[float, float, float] | None
    material: BackendResourceMaterial | None
    collision_mesh_path: str | None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        asset_id: str,
        backend: str,
    ) -> "BackendResourceAdapter":
        mesh_path = str(payload.get("mesh_path") or payload.get("relative_path") or "")
        resource_hash = str(payload.get("resource_hash") or payload.get("variant_hash") or "")
        if not resource_hash:
            raise ValueError(f"{asset_id}.{backend} backend resource requires resource_hash")
        material_payload = payload.get("material")
        mesh_scale = _optional_scale(payload.get("mesh_scale", payload.get("scale")))
        collision_mesh_path = payload.get("collision_mesh_path")
        return cls(
            asset_id=asset_id,
            backend=backend,
            resource_id=str(payload["resource_id"]) if payload.get("resource_id") else None,
            mesh_path=mesh_path,
            resource_hash=resource_hash,
            mesh_scale=mesh_scale,
            material=(
                BackendResourceMaterial.from_mapping(material_payload)
                if isinstance(material_payload, Mapping)
                else None
            ),
            collision_mesh_path=str(collision_mesh_path) if collision_mesh_path else None,
        )


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
            generated_files=tuple(str(path) for path in payload.get("generated_files", [])),
            preview_files=tuple(str(path) for path in payload.get("preview_files", [])),
        )


@dataclass(frozen=True, slots=True)
class CsdRealizationValidationRecord:
    """Validation summary for one realized backend manifest."""

    validation_id: str
    csd_id: str
    backend: str
    manifest_id: str
    cache_key: str
    status: str
    evidence_files: tuple[str, ...]
    preview_files: tuple[str, ...]
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.validation_id:
            raise ValueError("validation_id is required")
        if not self.csd_id:
            raise ValueError("csd_id is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.manifest_id:
            raise ValueError("manifest_id is required")
        if not self.cache_key:
            raise ValueError("cache_key is required")
        if self.status not in {"passed", "failed"}:
            raise ValueError("validation status must be 'passed' or 'failed'")
        object.__setattr__(
            self,
            "evidence_files",
            tuple(str(path) for path in self.evidence_files),
        )
        object.__setattr__(
            self,
            "preview_files",
            tuple(str(path) for path in self.preview_files),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "cache_key": self.cache_key,
            "csd_id": self.csd_id,
            "evidence_files": list(self.evidence_files),
            "manifest_id": self.manifest_id,
            "preview_files": list(self.preview_files),
            "schema_version": self.schema_version,
            "status": self.status,
            "validation_id": self.validation_id,
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CsdRealizationValidationRecord":
        return cls(
            validation_id=str(payload["validation_id"]),
            csd_id=str(payload["csd_id"]),
            backend=str(payload["backend"]),
            manifest_id=str(payload["manifest_id"]),
            cache_key=str(payload["cache_key"]),
            status=str(payload["status"]),
            evidence_files=tuple(str(path) for path in payload.get("evidence_files", [])),
            preview_files=tuple(str(path) for path in payload.get("preview_files", [])),
            schema_version=str(payload.get("schema_version", "0.1")),
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
    csd: Mapping[str, Any] | None = None,
    csd_hash: str | None = None,
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
    if (csd is None) == (csd_hash is None):
        raise ValueError("exactly one of csd or csd_hash is required")
    if csd_hash is not None:
        resolved_csd_hash = csd_hash
    else:
        assert csd is not None
        resolved_csd_hash = _canonical_hash(csd)
    asset_variant_hash = _canonical_hash(asset_variant_hashes)
    realization_config_hash = _canonical_hash(realization_config)
    digest = _canonical_hash(
        {
            "backend": backend,
            "csd_hash": resolved_csd_hash,
            "asset_variant_hash": asset_variant_hash,
            "realization_config_hash": realization_config_hash,
            "realization_version": realization_version,
            "simulator_version": simulator_version,
        }
    )
    return CsdRealizationCacheKey(
        backend=backend,
        digest=digest,
        csd_hash=resolved_csd_hash,
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
    resources = backend_resource_adapters_by_asset(asset_registry, backend)
    blockers: list[CsdRealizationBlocker] = []
    for asset_id in _csd_asset_ids(csd):
        if asset_id not in resources:
            blockers.append(
                CsdRealizationBlocker(
                    blocker_id=f"{csd_id}_{backend}_{asset_id}_resource_missing",
                    csd_id=csd_id,
                    backend=backend,
                    asset_id=asset_id,
                    scope="asset",
                    reason=f"asset has no backend resource adapter for {backend}",
                )
            )
    return tuple(blockers)


def asset_resource_hashes_for_csd(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    backend: str,
) -> dict[str, str]:
    """Return backend resource hashes for all CSD assets, after compatibility checks."""
    blockers = find_csd_realization_blockers(
        csd=csd,
        asset_registry=asset_registry,
        backend=backend,
    )
    if blockers:
        raise ValueError("CSD has unresolved realization blockers")
    resources = backend_resource_adapters_by_asset(asset_registry, backend)
    return {asset_id: resources[asset_id].resource_hash for asset_id in _csd_asset_ids(csd)}


def asset_variant_hashes_for_csd(
    *,
    csd: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    backend: str,
) -> dict[str, str]:
    """Backward-compatible alias for backend resource hashes."""
    return asset_resource_hashes_for_csd(
        csd=csd,
        asset_registry=asset_registry,
        backend=backend,
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_mapping(payload: Mapping[str, Any], key: str, *, field: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}.{key} must be a mapping")
    return value


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_list(value: object, *, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    entries: list[Mapping[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{field} entries must be mappings")
        entries.append(entry)
    return tuple(entries)


def _required_str(payload: Mapping[str, Any], key: str, *, field: str) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise ValueError(f"{field}.{key} is required")
    return value


def _required_float(payload: Mapping[str, Any], key: str, *, field: str) -> float:
    if key not in payload:
        raise ValueError(f"{field}.{key} is required")
    return float(payload[key])


def _number_tuple(value: object, *, length: int, field: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-element sequence")
    return tuple(float(item) for item in value)


def _friction_tuple(value: object) -> tuple[float, float, float]:
    if isinstance(value, (int, float, str)):
        return (float(value), DEFAULT_MUJOCO_OBJECT_FRICTION[1], DEFAULT_MUJOCO_OBJECT_FRICTION[2])
    friction = _number_tuple(value, length=3, field="initial_state.friction")
    return (float(friction[0]), float(friction[1]), float(friction[2]))


def _vector_from_sequence(value: object, *, field: str) -> CsdVector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a 3-element sequence")
    return CsdVector3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _identity_pose() -> CsdPose:
    return CsdPose(
        position=CsdVector3(0.0, 0.0, 0.0),
        orientation=CsdQuaternion(1.0, 0.0, 0.0, 0.0),
    )


def _csd_id(csd: Mapping[str, Any]) -> str:
    csd_id = str(csd.get("csd_id", ""))
    if not csd_id:
        raise ValueError("csd_id is required")
    return csd_id


def _csd_asset_ids(csd: Mapping[str, Any]) -> tuple[str, ...]:
    asset_ids: list[str] = []
    for obj in _scenario(csd).get("objects", []):
        if isinstance(obj, Mapping) and obj.get("asset_id"):
            asset_ids.append(str(obj["asset_id"]))
    return tuple(dict.fromkeys(asset_ids))


def _scenario(csd: Mapping[str, Any]) -> Mapping[str, Any]:
    scenario = csd.get("scenario")
    return scenario if isinstance(scenario, Mapping) else csd


def backend_resource_adapters_by_asset(
    asset_registry: Mapping[str, Any],
    backend: str,
) -> dict[str, BackendResourceAdapter]:
    """Return typed backend resource adapters keyed by accepted asset id."""
    resources: dict[str, BackendResourceAdapter] = {}
    for obj in asset_registry.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        asset_id = str(obj.get("asset_id") or obj.get("object_id") or "")
        if not asset_id:
            continue
        resource = _backend_resource_adapter(obj, backend)
        if resource is not None:
            resources[asset_id] = resource
    for asset in asset_registry.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        asset_id = str(asset.get("asset_id", ""))
        if not asset_id or asset_id in resources:
            continue
        resource = _backend_resource_adapter(asset, backend)
        if resource is not None:
            resources[asset_id] = resource
    return resources


def _backend_resource_adapter(
    record: Mapping[str, Any],
    backend: str,
) -> BackendResourceAdapter | None:
    entries = record.get("backend_resources", record.get("variants", ()))
    if not isinstance(entries, (list, tuple)):
        return None
    asset_id = str(record.get("asset_id") or record.get("object_id") or "")
    if not asset_id:
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("backend") or entry.get("engine") or "") == backend:
            return BackendResourceAdapter.from_mapping(
                entry,
                asset_id=asset_id,
                backend=backend,
            )
    return None


def _optional_scale(value: object) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError("mesh_scale must be a number or a 3-element sequence")
