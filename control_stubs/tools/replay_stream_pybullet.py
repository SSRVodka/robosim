"""Replay an instruction-stream artifact on the PyBullet backend.

Consumes the engine-agnostic (t, q[7], gripper) stream by simulated time:
the backend loop is paused and physics is stepped manually, one 1/fps frame
at a time (exactly 8 x 1/240s steps at 30 fps), so the replay is
deterministic and independent of wall-clock pacing. Success is judged by the
backend-local predicate (cup pose inside the container bounds) — task-level
semantic equivalence, not frame-level state equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from control_stubs import common_pb2
from control_stubs import robot_core_pb2 as core_pb2
from control_stubs.tools.traj_synthesis import ARM_JOINTS, InstructionStream
from robosim.backends import PyBulletBackend

FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GROUP = "panda_arm_hand"
BACKEND_TIMESTEP = 1.0 / 240.0
SETTLE_SEC = 1.0
CONTAINER_HALF_EXTENT = 0.15
CONTAINER_MIN_Z = 0.05


def _pose_to_array(pose) -> np.ndarray:  # noqa: ANN001 - proto message
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def _command(backend: PyBulletBackend, q: np.ndarray, gripper: float) -> None:
    backend.set_joint_target(
        ARM_JOINTS + FINGER_JOINTS,
        [float(value) for value in q] + [gripper, gripper],
        core_pb2.JointCommand.ControlMode.POSITION,
        GROUP,
    )


def _step(backend: PyBulletBackend, seconds: float) -> None:
    for _ in range(int(round(seconds / BACKEND_TIMESTEP))):
        backend.step_physics()


def replay(stream_path: Path, scene_path: Path, scene_meta_path: Path) -> dict[str, object]:
    stream = InstructionStream.read(stream_path)
    fps = int(stream.meta["fps"])
    steps_per_frame = round(1.0 / fps / BACKEND_TIMESTEP)
    if not np.isclose(steps_per_frame * BACKEND_TIMESTEP * fps, 1.0):
        raise ValueError(f"fps {fps} is not commensurate with 1/240s backend timestep")

    backend = PyBulletBackend(str(scene_path), str(scene_meta_path), headless=True)
    try:
        backend.pause()
        # Realize the episode initial state recorded at synthesis time: the
        # stream is only valid for the object layout it was generated for.
        object_positions = stream.meta.get("object_positions", {})
        for name, position in object_positions.items():
            if name == "container":
                continue  # static compound; scene fixture must already match
            backend.set_object_pose(
                name,
                common_pb2.Pose(
                    position=common_pb2.Point(
                        x=position[0], y=position[1], z=position[2]
                    ),
                    orientation=common_pb2.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
        first = stream.samples[0]
        _command(backend, first.q, first.gripper)
        _step(backend, SETTLE_SEC)  # align start state before replay

        tracking_error = 0.0
        arm_indices = [
            backend._joint_infos_by_name[name].index  # noqa: SLF001 - diagnostics only
            for name in ARM_JOINTS
        ]
        import pybullet as p  # noqa: PLC0415 - diagnostics only

        for sample in stream.samples:
            _command(backend, sample.q, sample.gripper)
            for _ in range(steps_per_frame):
                backend.step_physics()
            states = p.getJointStates(
                backend._robot_body_id,  # noqa: SLF001 - diagnostics only
                arm_indices,
                physicsClientId=backend._client_id,  # noqa: SLF001
            )
            achieved = np.array([state[0] for state in states])
            tracking_error = max(
                tracking_error, float(np.abs(achieved - sample.q).max())
            )

        _step(backend, SETTLE_SEC)
        cup = _pose_to_array(backend.get_object_pose("cup"))
        container = _pose_to_array(backend.get_object_pose("container"))
        success = bool(
            abs(cup[0] - container[0]) < CONTAINER_HALF_EXTENT
            and abs(cup[1] - container[1]) < CONTAINER_HALF_EXTENT
            and cup[2] > CONTAINER_MIN_Z
        )
        return {
            "backend": "pybullet",
            "stream": stream_path.name,
            "frames": len(stream.samples),
            "success": success,
            "cup_final": cup.tolist(),
            "container": container.tolist(),
            "max_arm_tracking_error_rad": tracking_error,
        }
    finally:
        backend.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "tests"
        / "fixtures"
        / "pybullet_cup_scene"
        / "scene.py",
    )
    parser.add_argument("--scene-meta", type=Path, default=None)
    args = parser.parse_args(argv)
    scene_meta = args.scene_meta or args.scene.with_name("scene_meta.json")
    result = replay(args.stream, args.scene, scene_meta)
    print(json.dumps(result, indent=1))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
