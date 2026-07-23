"""Scripted pick-and-place expert data collection over gRPC.

Drives a Franka Panda in the MuJoCo scene
``drivers_sim/mujoco/assets/robots/franka_panda/scene.xml`` (red box on the
table, green container) through reach -> grasp -> lift -> place waypoints and
records each episode as a LeRobotDataset. All control goes through the vsim
gRPC surface; MuJoCo bindings are used client-side for kinematics only.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from control_stubs import common_pb2
from control_stubs.robot_core_pb2 import JointCommand
from control_stubs.tools.client import RobosimClient

ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
ARM_GROUP = "panda_arm"
HAND_GROUP = "panda_hand"
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRASP_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
BOX_HOME = np.array([0.6, -0.2, 0.179])
CONTAINER_XY = np.array([0.4, 0.3])
LIFT_HAND_Z = 0.45
GRASP_HAND_Z = 0.33
PLACE_HAND_Z = 0.50
CARTESIAN_STEP = 0.01
SETTLE_FRAMES = 20
GRIPPER_DWELL_FRAMES = 15
BOX_RANDOM_RANGE = 0.05


class PandaKinematics:
    """Client-side kinematics of the Panda scene for IK; not a simulation."""

    def __init__(self, scene_xml: Path) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._data = mujoco.MjData(self._model)
        joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINTS
        ]
        self._qpos_adr = [self._model.jnt_qposadr[jid] for jid in joint_ids]
        self._dof_adr = [self._model.jnt_dofadr[jid] for jid in joint_ids]
        self._joint_range = self._model.jnt_range[joint_ids]
        self._hand_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def hand_position(self, q: np.ndarray) -> np.ndarray:
        for value, adr in zip(q, self._qpos_adr, strict=True):
            self._data.qpos[adr] = value
        mujoco.mj_forward(self._model, self._data)
        return self._data.xpos[self._hand_id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        initial_q: np.ndarray,
        max_iters: int = 100,
    ) -> np.ndarray:
        q = initial_q.copy()
        error = np.zeros(6)
        quat_conj = np.zeros(4)
        quat_diff = np.zeros(4)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        for _ in range(max_iters):
            for value, adr in zip(q, self._qpos_adr, strict=True):
                self._data.qpos[adr] = value
            mujoco.mj_forward(self._model, self._data)
            error[:3] = target_pos - self._data.xpos[self._hand_id]
            mujoco.mju_negQuat(quat_conj, self._data.xquat[self._hand_id])
            mujoco.mju_mulQuat(quat_diff, target_quat_wxyz, quat_conj)
            mujoco.mju_quat2Vel(error[3:], quat_diff, 1.0)
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                return q
            mujoco.mj_jacBody(self._model, self._data, jacp, jacr, self._hand_id)
            jac = np.vstack([jacp[:, self._dof_adr], jacr[:, self._dof_adr]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            q = np.clip(
                q + np.clip(delta, -0.2, 0.2),
                self._joint_range[:, 0],
                self._joint_range[:, 1],
            )
        raise ValueError(
            f"IK did not converge to {target_pos.tolist()}; residual {error.tolist()}"
        )


def build_episode_targets(
    kinematics: PandaKinematics,
    box_pos: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Return the (arm_q, gripper) target stream for one pick-and-place episode."""
    above_box = np.array([box_pos[0], box_pos[1], LIFT_HAND_Z])
    grasp = np.array([box_pos[0], box_pos[1], GRASP_HAND_Z])
    lift = above_box
    place = np.array([CONTAINER_XY[0], CONTAINER_XY[1], PLACE_HAND_Z])

    targets: list[tuple[np.ndarray, float]] = []
    current_q = HOME_Q

    def move_to(cartesian_target: np.ndarray, gripper: float) -> None:
        nonlocal current_q
        start = kinematics.hand_position(current_q)
        distance = float(np.linalg.norm(cartesian_target - start))
        steps = max(2, int(np.ceil(distance / CARTESIAN_STEP)))
        for step in range(1, steps + 1):
            waypoint = start + (cartesian_target - start) * step / steps
            current_q = kinematics.solve(waypoint, GRASP_QUAT_WXYZ, current_q)
            targets.append((current_q.copy(), gripper))

    def dwell(gripper: float, frames: int) -> None:
        for _ in range(frames):
            targets.append((current_q.copy(), gripper))

    move_to(above_box, GRIPPER_OPEN)
    move_to(grasp, GRIPPER_OPEN)
    dwell(GRIPPER_OPEN, SETTLE_FRAMES)
    dwell(GRIPPER_CLOSED, GRIPPER_DWELL_FRAMES)
    move_to(lift, GRIPPER_CLOSED)
    move_to(place, GRIPPER_CLOSED)
    dwell(GRIPPER_CLOSED, SETTLE_FRAMES)
    dwell(GRIPPER_OPEN, GRIPPER_DWELL_FRAMES)
    return targets


def collect_episode(
    client: RobosimClient,
    kinematics: PandaKinematics,
    *,
    seed: int,
    repo_name: str,
    task_text: str,
    control_fps: int,
    record_fps: int,
    randomize_box: bool,
    exclude_sensors: list[str],
) -> float:
    """Run one episode and return its wall-clock duration in seconds."""
    client.simulation.reset_world(seed=seed)
    time.sleep(0.5)

    box_pos = BOX_HOME.copy()
    if randomize_box:
        rng = np.random.default_rng(seed)
        box_pos[:2] += rng.uniform(-BOX_RANDOM_RANGE, BOX_RANDOM_RANGE, size=2)
        status = client.simulation.set_object_pose(
            "box", tuple(box_pos), (0.0, 0.0, 0.0, 1.0)
        )
        if status.code != common_pb2.STATUS_SUCCESS:
            raise RuntimeError(f"set_object_pose failed: {status.message}")
        time.sleep(0.3)

    targets = build_episode_targets(kinematics, box_pos)

    start_time = time.monotonic()
    job = client.robot_data.episode_start(
        repo_name,
        task_text=task_text,
        fps=record_fps,
        sensor_name_excluded=exclude_sensors,
    )
    if job.status.code != common_pb2.STATUS_SUCCESS:
        raise RuntimeError(f"episode_start failed: {job.status.message}")

    try:
        deadline = time.monotonic()
        for arm_q, gripper in targets:
            client.robot_core.set_joint_target(
                ARM_JOINTS,
                [float(v) for v in arm_q],
                JointCommand.ControlMode.POSITION,
                ARM_GROUP,
            )
            client.robot_core.set_joint_target(
                FINGER_JOINTS,
                [gripper, gripper],
                JointCommand.ControlMode.POSITION,
                HAND_GROUP,
            )
            deadline += 1.0 / control_fps
            time.sleep(max(0.0, deadline - time.monotonic()))
        status = client.robot_data.episode_end()
    except BaseException:
        client.robot_data.episode_cancel()
        raise
    if status.code != common_pb2.STATUS_SUCCESS:
        raise RuntimeError(f"episode_end failed: {status.message}")
    return time.monotonic() - start_time


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scene", type=Path, required=True, help="Panda scene.xml for kinematics")
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--task-text", default="pick the red box and place it into the container")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--control-fps", type=int, default=20)
    parser.add_argument("--record-fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-box", action="store_true")
    parser.add_argument(
        "--exclude-sensors",
        nargs="*",
        default=[],
        help="Sensor names to exclude from recording (e.g. wrist_camera)",
    )
    args = parser.parse_args(argv)

    kinematics = PandaKinematics(args.scene)
    client = RobosimClient(args.host, args.port)
    durations: list[float] = []
    try:
        for episode in range(args.episodes):
            duration = collect_episode(
                client,
                kinematics,
                seed=args.seed + episode,
                repo_name=args.repo_name,
                task_text=args.task_text,
                control_fps=args.control_fps,
                record_fps=args.record_fps,
                randomize_box=args.randomize_box,
                exclude_sensors=list(args.exclude_sensors),
            )
            durations.append(duration)
            print(f"episode {episode}: {duration:.1f}s", flush=True)
    finally:
        client.close()
    print(
        f"collected {len(durations)} episodes in {sum(durations):.1f}s "
        f"(mean {sum(durations) / len(durations):.1f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
