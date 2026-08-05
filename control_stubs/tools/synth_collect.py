"""Synthesized-trajectory expert data collection with rejection sampling.

Per episode: sample object pose -> synthesize + gate (traj_synthesis) ->
execute on the simulator via ExecuteJointTrajectory while recording ->
evaluate the task predicate (cup inside container for N consecutive polls,
via GetObjectPose) -> keep the episode on success, cancel it otherwise.
Gate failures and predicate failures are written as typed blocker artifacts;
accepted episodes get a quality-metadata line in the run manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from control_stubs import common_pb2
from control_stubs.tools.client import RobosimClient
from control_stubs.tools.traj_synthesis import (
    ARM_JOINTS,
    MinkIK,
    SynthesisConfig,
    SynthesisError,
    scene_hash,
    synthesize,
    validate,
)

FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
ARM_HAND_GROUP = "panda_arm_hand"
CUP_HOME = np.array([0.6, -0.2, 0.151])
OBJECT_RANDOM_RANGE = 0.05
PREDICATE_POLLS = 10
PREDICATE_INTERVAL_SEC = 0.1
CONTAINER_HALF_EXTENT = 0.15
CONTAINER_MIN_Z = 0.05


def _pose_to_array(pose) -> np.ndarray:  # noqa: ANN001 - proto message
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def cup_in_container(client: RobosimClient) -> tuple[bool, dict[str, object]]:
    """LIBERO-style predicate: satisfied for PREDICATE_POLLS consecutive polls."""
    reply = client.simulation.get_object_pose("container")
    if reply.status.code != common_pb2.STATUS_SUCCESS:
        raise RuntimeError(f"get_object_pose(container) failed: {reply.status.message}")
    container = _pose_to_array(reply.pose)
    last: list[float] = []
    for _ in range(PREDICATE_POLLS):
        reply = client.simulation.get_object_pose("cup")
        if reply.status.code != common_pb2.STATUS_SUCCESS:
            raise RuntimeError(f"get_object_pose(cup) failed: {reply.status.message}")
        cup = _pose_to_array(reply.pose)
        last = cup.tolist()
        inside = (
            abs(cup[0] - container[0]) < CONTAINER_HALF_EXTENT
            and abs(cup[1] - container[1]) < CONTAINER_HALF_EXTENT
            and cup[2] > CONTAINER_MIN_Z
        )
        if not inside:
            return False, {"cup": last, "container": container.tolist()}
        time.sleep(PREDICATE_INTERVAL_SEC)
    return True, {"cup": last, "container": container.tolist()}


def collect(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--task-text", default="pick the cup and place it into the container")
    parser.add_argument("--episodes", type=int, default=1, help="accepted episodes to collect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="default: 3x --episodes")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="output dir for streams/, blockers/, manifest.jsonl")
    parser.add_argument("--exclude-sensors", nargs="*", default=[])
    args = parser.parse_args(argv)

    max_attempts = args.max_attempts or args.episodes * 3
    config = SynthesisConfig()
    ik = MinkIK(args.scene, config)
    streams_dir = args.run_dir / "streams"
    blockers_dir = args.run_dir / "blockers"
    manifest_path = args.run_dir / "manifest.jsonl"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    client = RobosimClient(args.host, args.port)
    accepted = 0
    attempt = 0
    try:
        with manifest_path.open("a") as manifest:
            while accepted < args.episodes and attempt < max_attempts:
                seed = args.seed + attempt
                attempt += 1
                rng = np.random.default_rng(seed)
                cup_pos = CUP_HOME.copy()
                cup_pos[:2] += rng.uniform(-OBJECT_RANDOM_RANGE, OBJECT_RANDOM_RANGE, 2)

                client.simulation.reset_world(seed=seed)
                time.sleep(0.5)
                status = client.simulation.set_object_pose(
                    "cup", tuple(cup_pos), (0.0, 0.0, 0.0, 1.0)
                )
                if status.code != common_pb2.STATUS_SUCCESS:
                    raise RuntimeError(f"set_object_pose failed: {status.message}")
                time.sleep(0.3)

                container = _pose_to_array(
                    client.simulation.get_object_pose("container").pose
                )
                objects = {"cup": cup_pos, "container": container}
                try:
                    stream = synthesize(ik, objects, config=config, seed=seed)
                    validate(ik, stream, config=config, cup_pose=cup_pos)
                except SynthesisError as error:
                    blocker = blockers_dir / f"seed{seed:05d}_{error.payload['stage']}.json"
                    error.payload["details"]["seed"] = seed  # type: ignore[index]
                    error.write(blocker)
                    stage = error.payload["stage"]
                    print(f"seed {seed}: synthesis rejected ({stage}) -> {blocker}")
                    continue

                stream.meta["scene_hash"] = scene_hash(args.scene)
                stream_path = streams_dir / f"seed{seed:05d}.jsonl"
                stream.write(stream_path)

                job = client.robot_data.episode_start(
                    args.repo_name,
                    task_text=args.task_text,
                    fps=stream.meta["fps"],
                    sensor_name_excluded=list(args.exclude_sensors),
                )
                if job.status.code != common_pb2.STATUS_SUCCESS:
                    raise RuntimeError(f"episode_start failed: {job.status.message}")
                try:
                    points = [
                        (sample.t, [float(v) for v in sample.q] + [sample.gripper] * 2)
                        for sample in stream.samples
                    ]
                    status = client.robot_core.execute_joint_trajectory(
                        ARM_JOINTS + FINGER_JOINTS, points, ARM_HAND_GROUP
                    )
                    if status.code != common_pb2.STATUS_SUCCESS:
                        raise RuntimeError(
                            f"execute_joint_trajectory failed: {status.message}"
                        )
                    success, evidence = cup_in_container(client)
                except BaseException:
                    client.robot_data.episode_cancel()
                    raise

                if success:
                    status = client.robot_data.episode_end()
                    if status.code != common_pb2.STATUS_SUCCESS:
                        raise RuntimeError(f"episode_end failed: {status.message}")
                    accepted += 1
                    manifest.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "stream": stream_path.name,
                                "cup_pos": cup_pos.tolist(),
                                "duration": stream.samples[-1].t,
                                "samples": len(stream.samples),
                                "ik_worst_residual": stream.meta["ik_worst_residual"],
                                "predicate": evidence,
                            }
                        )
                        + "\n"
                    )
                    manifest.flush()
                    print(f"seed {seed}: accepted ({accepted}/{args.episodes})")
                else:
                    client.robot_data.episode_cancel()
                    blocker = blockers_dir / f"seed{seed:05d}_predicate.json"
                    blocker.parent.mkdir(parents=True, exist_ok=True)
                    blocker.write_text(
                        json.dumps(
                            {
                                "schema": "synthesis_blocker/v1",
                                "stage": "task_predicate",
                                "reason": "cup not inside container after execution",
                                "details": {"seed": seed, **evidence},
                            },
                            indent=1,
                        )
                    )
                    print(f"seed {seed}: predicate failed -> {blocker}")
    finally:
        client.close()

    print(f"accepted {accepted}/{args.episodes} episodes in {attempt} attempts")
    return 0 if accepted == args.episodes else 1


if __name__ == "__main__":
    sys.exit(collect())
