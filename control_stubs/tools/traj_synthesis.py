"""Object-frame trajectory synthesis for scripted expert data collection.

Pipeline: object poses -> semantic keypose stream (object-frame offsets) ->
per-segment Cartesian time parameterization (ruckig, state-to-state only) ->
per-sample differential IK (mink FrameTask + posture anchor) -> validation
gate (IK convergence, joint limits, per-step velocity, phase-aware kinematic
contact scan) -> instruction-stream JSONL artifact or typed blocker artifact.

ruckig red line: the community build silently offloads problems that set
``intermediate_positions`` to a cloud API, so this module only ever uses
single-segment state-to-state queries. Cartesian-space parameterization keeps
the hand on a straight line during grasp descent; joint-space interpolation
between keyposes would bow the hand path into the cup rim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
from ruckig import InputParameter, Result, Ruckig, Trajectory

HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRASP_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]

# Official Franka Panda joint limits (frankaemika.github.io/docs/control_parameters.html).
PANDA_VELOCITY_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

SCHEMA_STREAM = "instruction_stream/v1"
SCHEMA_BLOCKER = "synthesis_blocker/v1"


@dataclass(frozen=True)
class Keypose:
    """Semantic keypose: hand offset in an object frame plus gripper command."""

    name: str
    frame: str  # object body name, or "world"
    offset: tuple[float, float, float]
    gripper: float
    dwell_after: float = 0.0


# Same semantic sequence as scripted_collect.build_episode_targets: heights are
# hand-frame z targets expressed as offsets from the cup (z=0.151) and
# container (z=0.15) body origins.
PICK_PLACE_KEYPOSES = [
    Keypose("above_cup", "cup", (0.0, 0.0, 0.299), GRIPPER_OPEN),
    Keypose("grasp", "cup", (0.0, 0.0, 0.129), GRIPPER_OPEN, dwell_after=1.0),
    Keypose("grasp_close", "cup", (0.0, 0.0, 0.129), GRIPPER_CLOSED, dwell_after=1.0),
    Keypose("lift", "cup", (0.0, 0.0, 0.299), GRIPPER_CLOSED),
    Keypose("place", "container", (0.0, 0.0, 0.35), GRIPPER_CLOSED, dwell_after=1.0),
    Keypose("release", "container", (0.0, 0.0, 0.35), GRIPPER_OPEN, dwell_after=1.0),
]
GRASP_CLOSE_NAME = "grasp_close"


@dataclass(frozen=True)
class SynthesisConfig:
    fps: int = 30
    cartesian_vmax: float = 0.25  # m/s
    cartesian_amax: float = 1.0  # m/s^2
    cartesian_jmax: float = 5.0  # m/s^3
    ik_max_iters: int = 200
    ik_pos_tol: float = 5e-4  # m
    ik_ori_tol: float = 2e-3  # rad
    joint_velocity_scale: float = 0.8  # gate margin on official limits
    joint_range_margin: float = 0.01  # rad


DEFAULT_CONFIG = SynthesisConfig()


@dataclass
class Sample:
    t: float
    q: np.ndarray  # (7,) arm joint positions
    gripper: float


@dataclass
class InstructionStream:
    samples: list[Sample]
    meta: dict[str, object]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write(json.dumps({"schema": SCHEMA_STREAM, **self.meta}) + "\n")
            for sample in self.samples:
                fh.write(
                    json.dumps(
                        {
                            "t": round(sample.t, 6),
                            "q": [round(float(v), 6) for v in sample.q],
                            "g": sample.gripper,
                        }
                    )
                    + "\n"
                )

    @staticmethod
    def read(path: Path) -> "InstructionStream":
        with path.open() as fh:
            meta = json.loads(fh.readline())
            if meta.get("schema") != SCHEMA_STREAM:
                raise ValueError(f"unexpected schema in {path}: {meta.get('schema')}")
            samples = [
                Sample(row["t"], np.asarray(row["q"]), row["g"])
                for row in map(json.loads, fh)
            ]
        return InstructionStream(samples, meta)


class SynthesisError(Exception):
    """Trajectory synthesis failure carrying a typed blocker payload."""

    def __init__(self, stage: str, reason: str, details: dict[str, object]):
        super().__init__(f"{stage}: {reason}")
        self.payload: dict[str, object] = {
            "schema": SCHEMA_BLOCKER,
            "stage": stage,
            "reason": reason,
            "details": details,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=1))


class MinkIK:
    """Differential IK on the scene model: FrameTask on hand + posture anchor."""

    def __init__(self, scene_xml: Path, config: SynthesisConfig) -> None:
        self._config = config
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINTS
        ]
        self.qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in joint_ids])
        self.dof_adr = np.array([self.model.jnt_dofadr[jid] for jid in joint_ids])
        self.joint_range = self.model.jnt_range[joint_ids].copy()
        self._hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._configuration = mink.Configuration(self.model)
        self._frame_task = mink.FrameTask(
            frame_name="hand",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.5,
        )
        self._posture_task = mink.PostureTask(self.model, cost=1e-2)
        self._arm_dof_mask = np.zeros(self.model.nv, dtype=bool)
        self._arm_dof_mask[self.dof_adr] = True

    def _set_arm(self, q: np.ndarray) -> None:
        qpos = self._configuration.q.copy()
        qpos[self.qpos_adr] = q
        self._configuration.update(qpos)

    def hand_position(self, q: np.ndarray) -> np.ndarray:
        self._set_arm(q)
        return self._configuration.data.xpos[self._hand_id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        initial_q: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return (q, pos_err, ori_err); caller checks tolerances."""
        config = self._config
        self._set_arm(initial_q)
        self._posture_task.set_target_from_configuration(self._configuration)
        self._frame_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(np.asarray(target_quat_wxyz, dtype=float)), target_pos
            )
        )
        dt = 5e-3
        quat_conj = np.zeros(4)
        quat_diff = np.zeros(4)
        ori_vec = np.zeros(3)
        pos_err = ori_err = np.inf
        for _ in range(config.ik_max_iters):
            data = self._configuration.data
            pos_err = float(np.linalg.norm(target_pos - data.xpos[self._hand_id]))
            mujoco.mju_negQuat(quat_conj, data.xquat[self._hand_id])
            mujoco.mju_mulQuat(quat_diff, target_quat_wxyz, quat_conj)
            mujoco.mju_quat2Vel(ori_vec, quat_diff, 1.0)
            ori_err = float(np.linalg.norm(ori_vec))
            if pos_err < config.ik_pos_tol and ori_err < config.ik_ori_tol:
                break
            velocity = mink.solve_ik(
                self._configuration,
                [self._frame_task, self._posture_task],
                dt,
                solver="daqp",
                damping=1e-3,
            )
            velocity[~self._arm_dof_mask] = 0.0
            self._configuration.integrate_inplace(velocity, dt)
        return self._configuration.q[self.qpos_adr].copy(), pos_err, ori_err


def _cartesian_segment(start: np.ndarray, end: np.ndarray, config: SynthesisConfig) -> Trajectory:
    inp = InputParameter(3)
    inp.current_position = list(start)
    inp.target_position = list(end)
    inp.max_velocity = [config.cartesian_vmax] * 3
    inp.max_acceleration = [config.cartesian_amax] * 3
    inp.max_jerk = [config.cartesian_jmax] * 3
    trajectory = Trajectory(3)
    result = Ruckig(3).calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise SynthesisError(
            "time_parameterization",
            f"ruckig returned {result}",
            {"start": start.tolist(), "end": end.tolist()},
        )
    return trajectory


def _resolve_keypose(pose: Keypose, object_positions: dict[str, np.ndarray]) -> np.ndarray:
    if pose.frame == "world":
        return np.asarray(pose.offset, dtype=float)
    if pose.frame not in object_positions:
        raise SynthesisError(
            "keypose_resolution",
            f"no pose for object '{pose.frame}'",
            {"keypose": pose.name},
        )
    return object_positions[pose.frame] + np.asarray(pose.offset, dtype=float)


def synthesize(
    ik: MinkIK,
    object_positions: dict[str, np.ndarray],
    *,
    config: SynthesisConfig = DEFAULT_CONFIG,
    keyposes: list[Keypose] = PICK_PLACE_KEYPOSES,
    seed: int | None = None,
) -> InstructionStream:
    """Synthesize one episode; raises SynthesisError with a typed payload."""
    samples: list[Sample] = []
    ik_worst = {"pos": 0.0, "ori": 0.0}
    current_q = HOME_Q.copy()
    t = 0.0
    step = 1.0 / config.fps

    def emit(q: np.ndarray, gripper: float) -> None:
        nonlocal t
        t += step
        samples.append(Sample(t, q.copy(), gripper))

    for pose in keyposes:
        target = _resolve_keypose(pose, object_positions)
        start = ik.hand_position(current_q)
        distance = float(np.linalg.norm(target - start))
        if distance > 1e-6:
            trajectory = _cartesian_segment(start, target, config)
            frames = max(1, int(np.ceil(trajectory.duration * config.fps)))
            for frame in range(1, frames + 1):
                at = min(frame * step, trajectory.duration)
                waypoint = np.asarray(trajectory.at_time(at)[0])
                current_q, pos_err, ori_err = ik.solve(
                    waypoint, GRASP_QUAT_WXYZ, current_q
                )
                ik_worst["pos"] = max(ik_worst["pos"], pos_err)
                ik_worst["ori"] = max(ik_worst["ori"], ori_err)
                if pos_err > config.ik_pos_tol or ori_err > config.ik_ori_tol:
                    raise SynthesisError(
                        "ik_convergence",
                        f"IK residual pos={pos_err:.2e} ori={ori_err:.2e} at {pose.name}",
                        {"keypose": pose.name, "waypoint": waypoint.tolist(), "seed": seed},
                    )
                emit(current_q, pose.gripper)
        for _ in range(int(round(pose.dwell_after * config.fps))):
            emit(current_q, pose.gripper)

    meta = {
        "fps": config.fps,
        "seed": seed,
        "object_positions": {k: v.tolist() for k, v in object_positions.items()},
        "keyposes": [pose.name for pose in keyposes],
        "ik_worst_residual": ik_worst,
        "versions": {
            name: importlib.metadata.version(name) for name in ("mujoco", "mink", "ruckig")
        },
    }
    return InstructionStream(samples, meta)


def _robot_geoms(model: mujoco.MjModel) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    robot_bodies = {
        body
        for body in range(model.nbody)
        if _has_ancestor(model, body, root)
    }
    return {
        geom for geom in range(model.ngeom) if model.geom_bodyid[geom] in robot_bodies
    }


def _has_ancestor(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body != 0:
        if body == ancestor:
            return True
        body = model.body_parentid[body]
    return False


def _object_geoms(model: mujoco.MjModel, body_names: list[str]) -> set[int]:
    geoms: set[int] = set()
    for name in body_names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms.update(
            geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body
        )
    return geoms


def validate(
    ik: MinkIK,
    stream: InstructionStream,
    *,
    config: SynthesisConfig = DEFAULT_CONFIG,
    cup_pose: np.ndarray | None = None,
) -> None:
    """Gate the stream; raises SynthesisError on the first violated check."""
    q_matrix = np.stack([sample.q for sample in stream.samples])
    low = ik.joint_range[:, 0] + config.joint_range_margin
    high = ik.joint_range[:, 1] - config.joint_range_margin
    if ((q_matrix < low) | (q_matrix > high)).any():
        index = int(np.argwhere((q_matrix < low) | (q_matrix > high))[0][0])
        raise SynthesisError(
            "joint_range",
            f"joint range violated at sample {index}",
            {"sample": index, "q": q_matrix[index].tolist()},
        )
    step_velocity = np.abs(np.diff(q_matrix, axis=0)) * config.fps
    limit = PANDA_VELOCITY_LIMIT * config.joint_velocity_scale
    if (step_velocity > limit).any():
        index = int(np.argwhere((step_velocity > limit).any(axis=1))[0][0])
        raise SynthesisError(
            "joint_velocity",
            f"per-step velocity exceeds {config.joint_velocity_scale}x official limit",
            {"sample": index, "velocity": step_velocity[index].tolist()},
        )

    # Phase-aware kinematic contact scan on a scratch copy of the scene.
    model = ik.model
    data = mujoco.MjData(model)
    if cup_pose is not None:
        cup_joint = model.body_jntadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        ]
        adr = model.jnt_qposadr[cup_joint]
        data.qpos[adr : adr + 3] = cup_pose
        data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    robot = _robot_geoms(model)
    close_index = next(
        (
            index
            for index, sample in enumerate(stream.samples)
            if sample.gripper == GRIPPER_CLOSED
        ),
        len(stream.samples),
    )
    finger_qpos = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    pre_close_forbidden = _object_geoms(model, ["cup", "table", "container"])
    post_close_forbidden = _object_geoms(model, ["table", "container"])
    for index, sample in enumerate(stream.samples):
        data.qpos[ik.qpos_adr] = sample.q
        for adr in finger_qpos:
            data.qpos[adr] = sample.gripper
        mujoco.mj_forward(model, data)
        forbidden = pre_close_forbidden if index < close_index else post_close_forbidden
        for contact in data.contact[: data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & robot and pair & forbidden:
                names = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    or mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                    )
                    for geom in sorted(pair)
                ]
                phase = "pre" if index < close_index else "post"
                raise SynthesisError(
                    "collision",
                    f"contact {names} at sample {index} ({phase}-grasp)",
                    {"sample": index, "geoms": names},
                )


def scene_hash(scene_xml: Path) -> str:
    return hashlib.sha256(scene_xml.read_bytes()).hexdigest()[:16]
