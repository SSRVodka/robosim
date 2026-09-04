"""CSD realization cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class CsdQuaternion:
    """Quaternion stored in MuJoCo-compatible wxyz order."""

    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class CsdPose:
    """Pose in the CSD root frame."""

    position: CsdVector3
    orientation: CsdQuaternion


@dataclass(frozen=True, slots=True)
class CsdSurface:
    """Backend-neutral static environment surface."""

    surface_id: str
    surface_type: str
    pose: CsdPose
    size: CsdVector3
    rgba: tuple[float, float, float, float]
    friction: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CsdCamera:
    """CSD camera requested by a scenario."""

    camera_id: str
    position: CsdVector3
    xyaxes: tuple[float, float, float, float, float, float] | None
    mode: str


@dataclass(frozen=True, slots=True)
class CsdLight:
    """CSD light source."""

    light_id: str
    position: CsdVector3
    direction: CsdVector3


@dataclass(frozen=True, slots=True)
class CsdEnvironment:
    """Global background/environment portion of a CSD scenario."""

    environment_id: str
    environment_type: str
    gravity: CsdVector3
    surfaces: tuple[CsdSurface, ...]
    cameras: tuple[CsdCamera, ...]
    lighting: tuple[CsdLight, ...]


@dataclass(frozen=True, slots=True)
class CsdRobot:
    """Robot instance requested by a CSD scenario."""

    asset_id: str
    pose: CsdPose


@dataclass(frozen=True, slots=True)
class CsdObjectContact:
    """Optional backend-neutral contact parameters for an object geom."""

    margin_m: float | None
    gap_m: float | None
    solref: tuple[float, float] | None
    solimp: tuple[float, float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class CsdObjectInertial:
    """Optional explicit object inertia in the object body frame."""

    center_of_mass: CsdVector3
    diagonal_inertia_kg_m2: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CsdObjectInitialState:
    """Backend-neutral object physical state requested by CSD."""

    mass_kg: float
    friction: tuple[float, float, float]
    contact: CsdObjectContact | None
    inertial: CsdObjectInertial | None


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


@dataclass(frozen=True, slots=True)
class CsdRelationship:
    """Compiler-readable CSD relationship."""

    relation_id: str
    type: CsdRelationshipType
    subject: str
    object: str
    parameters: Mapping[str, Any]


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
class CsdGazeboRuntimeContract:
    """Package-local ROS 2 endpoints required by a Gazebo realization."""

    world_file: str
    robot_control_urdf: str
    controllers_file: str
    namespace: str
    joint_state_topic: str
    trajectory_action: str
    camera_topics: Mapping[str, str]
    gripper_trajectory_action: str = ""
    robot_semantics_file: str = ""
    trajectory_actions: Mapping[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "world_file": self.world_file,
            "robot_control_urdf": self.robot_control_urdf,
            "controllers_file": self.controllers_file,
            "namespace": self.namespace,
            "joint_state_topic": self.joint_state_topic,
            "trajectory_action": self.trajectory_action,
            "camera_topics": dict(self.camera_topics),
            "gripper_trajectory_action": self.gripper_trajectory_action,
            "robot_semantics_file": self.robot_semantics_file,
            "trajectory_actions": dict(self.trajectory_actions),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CsdGazeboRuntimeContract":
        return cls(
            world_file=str(payload["world_file"]),
            robot_control_urdf=str(payload["robot_control_urdf"]),
            controllers_file=str(payload["controllers_file"]),
            namespace=str(payload["namespace"]),
            joint_state_topic=str(payload["joint_state_topic"]),
            trajectory_action=str(payload["trajectory_action"]),
            camera_topics={
                str(key): str(value) for key, value in dict(payload["camera_topics"]).items()
            },
            gripper_trajectory_action=str(payload.get("gripper_trajectory_action", "")),
            robot_semantics_file=str(payload.get("robot_semantics_file", "")),
            trajectory_actions={
                str(key): str(value)
                for key, value in dict(payload.get("trajectory_actions", {})).items()
            },
        )


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
    initial_state_file: str | None = None
    gazebo_runtime: CsdGazeboRuntimeContract | None = None

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
            "initial_state_file": self.initial_state_file,
            "gazebo_runtime": (
                self.gazebo_runtime.to_json_dict() if self.gazebo_runtime is not None else None
            ),
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
            initial_state_file=(
                str(payload["initial_state_file"])
                if payload.get("initial_state_file") is not None
                else None
            ),
            gazebo_runtime=(
                CsdGazeboRuntimeContract.from_json_dict(payload["gazebo_runtime"])
                if isinstance(payload.get("gazebo_runtime"), Mapping)
                else None
            ),
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
    csd_hash: str,
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
