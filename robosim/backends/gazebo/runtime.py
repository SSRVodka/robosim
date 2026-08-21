"""Server-owned lifecycle for one Gazebo realization package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from robosim.core.csd import CsdGazeboRuntimeContract, CsdRealizationManifest
from robosim.core.ros_environment import flush_ros_environment


def validate_runtime_contract(
    root: Path, runtime: CsdGazeboRuntimeContract
) -> dict[str, Any]:
    """Validate the complete package-local contract before starting processes."""
    artifacts = (
        runtime.world_file,
        runtime.robot_control_urdf,
        runtime.controllers_file,
        runtime.robot_semantics_file,
    )
    if any(not item or Path(item).is_absolute() or ".." in Path(item).parts for item in artifacts):
        raise ValueError("Gazebo runtime contract contains a non-local artifact path")
    if any(not (root / item).is_file() for item in artifacts):
        raise ValueError("Gazebo runtime contract artifact is missing")
    semantics = json.loads((root / runtime.robot_semantics_file).read_text(encoding="utf-8"))
    groups = dict(semantics.get("controller_groups", {}))
    if not groups or set(groups) != set(runtime.trajectory_actions):
        raise ValueError("Gazebo runtime contract has inconsistent controller actions")
    return semantics


class GazeboRuntime:
    """Own Gazebo Classic and ROS controller processes for one manifest."""

    def __init__(self, manifest_path: Path, *, headless: bool) -> None:
        self._root = manifest_path.resolve().parent
        manifest = CsdRealizationManifest.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if manifest.backend != "gazebo" or manifest.gazebo_runtime is None:
            raise ValueError("Gazebo runtime requires a Gazebo realization manifest")
        self.contract = manifest.gazebo_runtime
        self.semantics = validate_runtime_contract(self._root, self.contract)
        self._headless = headless
        self._processes: list[subprocess.Popen[bytes]] = []

    def start(self) -> None:
        """Start the simulator and return after all controllers are active."""
        if self._processes:
            raise RuntimeError("Gazebo runtime is already started")
        runtime = self.contract
        env = self._environment()
        ros2 = str(Path(sys.prefix) / "bin" / "ros2")
        commands = (
            (
                ros2,
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                str(self._root / runtime.robot_control_urdf),
                "--ros-args",
                "-r",
                f"__ns:={runtime.namespace}",
            ),
            (str(Path(sys.prefix) / "bin" / "gzserver"), "--verbose", runtime.world_file),
        )
        try:
            self._processes.extend(
                subprocess.Popen(command, cwd=self._root, env=env, start_new_session=True)
                for command in commands
            )
            for controller in (
                "joint_state_broadcaster",
                *(
                    str(self.semantics["controller_names"][group])
                    for group in runtime.trajectory_actions
                ),
            ):
                result = subprocess.run(
                    (
                        ros2,
                        "run",
                        "controller_manager",
                        "spawner",
                        controller,
                        "-c",
                        f"{runtime.namespace}/controller_manager",
                    ),
                    cwd=self._root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"failed to spawn controller '{controller}': {result.stderr.strip()}"
                    )
            if not self._headless:
                self._processes.append(
                    subprocess.Popen(
                        (str(Path(sys.prefix) / "bin" / "gzclient"),),
                        cwd=self._root,
                        env=env,
                        start_new_session=True,
                    )
                )
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop viewer, simulator, then robot-state publisher."""
        for process in reversed(self._processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._processes.clear()

    def _environment(self) -> dict[str, str]:
        env = flush_ros_environment()
        logs = self._root / "runtime" / "logs"
        for name in ("ros", "gazebo"):
            (logs / name).mkdir(parents=True, exist_ok=True)
        env["ROS_LOG_DIR"] = str(logs / "ros")
        env["GAZEBO_LOG_PATH"] = str(logs / "gazebo")
        env["GAZEBO_RESOURCE_PATH"] = os.pathsep.join(
            filter(None, (str(self._root), env.get("GAZEBO_RESOURCE_PATH")))
        )
        return env
