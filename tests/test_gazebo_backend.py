from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosim.backends.gazebo.backend import GazeboBackend
from robosim.core.capabilities import Capability
from robosim.core.csd import CsdGazeboRuntimeContract, CsdRealizationManifest


class DummyLogger:
    def error(self, message: str) -> None:
        del message

    def warn(self, message: str) -> None:
        del message


class ResetOnlyGazeboBackend(GazeboBackend):
    def __init__(self) -> None:
        self._capabilities = Capability.SIMULATION_CONTROL

    def get_logger(self) -> DummyLogger:
        return DummyLogger()


def test_gazebo_reset_reports_unimplemented() -> None:
    backend = ResetOnlyGazeboBackend()

    with pytest.raises(NotImplementedError, match="Reset world"):
        backend.reset_world(seed=0, randomization_params={})


def test_manifest_control_requires_the_exact_declared_group() -> None:
    backend = object.__new__(GazeboBackend)
    backend._runtime = CsdGazeboRuntimeContract(
        world_file="world.sdf",
        robot_control_urdf="robot.urdf",
        controllers_file="controllers.yaml",
        namespace="/robosim/test",
        joint_state_topic="/robosim/test/joint_states",
        trajectory_action="/robosim/test/arm/follow_joint_trajectory",
        camera_topics={},
    )
    backend._robot_semantics = {"controller_groups": {"arm": ["joint1", "joint2"]}}
    client = object()
    backend._trajectory_clients = {"arm": client}

    assert backend._trajectory_controller("arm", ["joint2", "joint1"]) is client
    with pytest.raises(ValueError, match="explicit controller group"):
        backend._trajectory_controller(None, ["joint1", "joint2"])
    with pytest.raises(ValueError, match="exactly match"):
        backend._trajectory_controller("arm", ["joint1"])


def test_manifest_robot_spec_maps_srdf_end_effector_fields() -> None:
    backend = object.__new__(GazeboBackend)
    backend._runtime = CsdGazeboRuntimeContract(
        world_file="world.sdf",
        robot_control_urdf="robot.urdf",
        controllers_file="controllers.yaml",
        namespace="/robosim/test",
        joint_state_topic="/robosim/test/joint_states",
        trajectory_action="/robosim/test/arm/follow_joint_trajectory",
        camera_topics={},
    )
    backend._robot_semantics = {
        "robot_name": "robot",
        "joint_limits": {
            "joint1": {
                "type": "revolute",
                "lower": -1.0,
                "upper": 1.0,
                "velocity": 2.0,
                "effort": 3.0,
            }
        },
        "groups": {"arm": ["joint1"], "hand": []},
        "named_states": {},
        "end_effectors": [
            {
                "name": "tool",
                "parent_group": "arm",
                "group": "hand",
                "parent_link": "tool_link",
            }
        ],
    }

    spec = backend.get_robot_spec()

    assert spec.joint_model_groups[0].end_effectors[0].group_name == "hand"


def test_manifest_runtime_rejects_non_local_artifacts(tmp_path: Path) -> None:
    runtime = CsdGazeboRuntimeContract(
        world_file="/tmp/world.sdf",
        robot_control_urdf="robot.urdf",
        controllers_file="controllers.yaml",
        namespace="/robosim/test",
        joint_state_topic="/robosim/test/joint_states",
        trajectory_action="/robosim/test/arm/follow_joint_trajectory",
        camera_topics={},
        robot_semantics_file="runtime/robot_semantics.json",
        trajectory_actions={"arm": "/robosim/test/arm/follow_joint_trajectory"},
    )
    manifest = CsdRealizationManifest(
        manifest_id="gazebo-test",
        csd_id="test",
        backend="gazebo",
        cache_key="cache",
        root_path=str(tmp_path),
        entry_file="world.sdf",
        generated_files=(),
        preview_files=(),
        gazebo_runtime=runtime,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_json_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="non-local artifact"):
        GazeboBackend.from_csd_realization_manifest_file(path)
