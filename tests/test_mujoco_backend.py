"""Tests for the MuJoCo backend."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copytree

import mujoco
import numpy as np
import pytest

from control_stubs import common_pb2, sensing_pb2
from control_stubs import robot_core_pb2 as core_pb2
from robosim.backends.mujoco.backend import MuJoCoBackend
from robosim.core import CsdRealizationManifest, compile_csd_to_mujoco

SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/franka_panda/scene.xml"
)
G1_29DOF_SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "drivers_sim/mujoco/assets/robots/unitree_g1/g1_29dof.xml"
)
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "csd"


@pytest.fixture
def backend() -> Generator[MuJoCoBackend, None, None]:
    instance = MuJoCoBackend(str(SCENE_PATH), headless=True)
    try:
        yield instance
    finally:
        instance.shutdown()


def _wait_for_condition(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _load_json_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _fixture_mesh_half_extents(path: Path) -> tuple[float, float, float]:
    name = path.stem
    if "tray" in name:
        return (0.08, 0.055, 0.012)
    if "marker" in name:
        return (0.018, 0.018, 0.055)
    if "can" in name:
        return (0.035, 0.035, 0.08)
    if "mug" in name:
        return (0.035, 0.035, 0.055)
    return (0.035, 0.035, 0.035)


def _write_box_mesh(path: Path) -> None:
    hx, hy, hz = _fixture_mesh_half_extents(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"v {-hx} {-hy} {-hz}",
                f"v {hx} {-hy} {-hz}",
                f"v {hx} {hy} {-hz}",
                f"v {-hx} {hy} {-hz}",
                f"v {-hx} {-hy} {hz}",
                f"v {hx} {-hy} {hz}",
                f"v {hx} {hy} {hz}",
                f"v {-hx} {hy} {hz}",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            )
        ),
        encoding="utf-8",
    )


def _write_fixture_asset_files(asset_root: Path, asset_registry: dict[str, object]) -> None:
    records = asset_registry.get("objects", ())
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        resources = record.get("backend_resources", ())
        if not isinstance(resources, list):
            continue
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            mesh_path = resource.get("mesh_path") or resource.get("relative_path")
            if mesh_path:
                _write_box_mesh(asset_root / str(mesh_path))


def _image_array(image: sensing_pb2.CameraImage) -> np.ndarray:
    return np.frombuffer(image.data, dtype=np.uint8).reshape(
        image.height,
        image.width,
        3,
    )


def test_robot_spec_uses_srdf_groups(backend: MuJoCoBackend) -> None:
    spec = backend.get_robot_spec()

    assert spec.robot_name == "panda"
    groups = {group.name: group for group in spec.joint_model_groups}
    assert set(groups) == {"panda_arm", "panda_hand", "panda_arm_hand"}
    assert list(groups["panda_arm"].joint_names) == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    assert [ee.name for ee in groups["panda_arm"].end_effectors] == ["hand"]


def test_backend_loads_compiled_csd_realization_manifest(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    source_template = Path(__file__).resolve().parents[1] / (
        "drivers_sim/mujoco/assets/robots/franka_panda"
    )
    template_copy = tmp_path / "template_src" / "franka_panda"
    copytree(source_template, template_copy)
    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
        realization_config={"robot_template_dir": str(template_copy)},
    )
    assert isinstance(result.manifest, CsdRealizationManifest)

    instance = MuJoCoBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        spec = instance.get_robot_spec()
        sensors = {entry.name for entry in instance.list_sensors().entries}

        assert spec.robot_name == "panda"
        assert "panda_arm" in {group.name for group in spec.joint_model_groups}
        assert "world_camera" in sensors
        assert instance._model.body("mug").name == "mug"
        assert instance._model.body("surface_tabletop").name == "surface_tabletop"
    finally:
        instance.shutdown()


def test_backend_loads_compiled_csd_realization_manifest_file(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )
    assert isinstance(result.manifest, CsdRealizationManifest)
    manifest_path = (
        tmp_path / "engine_manifests" / "mujoco" / "csd_tabletop_0001" / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["root_path"] = "/stale/package/location"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    instance = MuJoCoBackend.from_csd_realization_manifest_file(manifest_path, headless=True)
    try:
        assert instance.robot_name == "panda"
        assert instance._model.body("mug").name == "mug"
    finally:
        instance.shutdown()


def test_backend_runtime_renders_and_steps_compiled_csd_realization(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    csd = _load_json_fixture("franka_tabletop_single_object.json")
    asset_registry = _load_json_fixture("asset_registry_mujoco.json")
    _write_fixture_asset_files(asset_root, asset_registry)
    result = compile_csd_to_mujoco(
        csd=csd,
        asset_registry=asset_registry,
        output_root=tmp_path / "engine_manifests",
        asset_root=asset_root,
    )
    assert isinstance(result.manifest, CsdRealizationManifest)

    instance = MuJoCoBackend.from_csd_realization_manifest(result.manifest, headless=True)
    try:
        image = instance.get_sensors(["world_camera"]).images[0]
        first_qpos = instance._data.qpos.copy()
        assert image.width == 320
        assert image.height == 240
        assert float(_image_array(image).std()) > 1.0

        time.sleep(0.05)

        assert np.isfinite(instance._data.qpos).all()
        assert np.isfinite(instance._data.qvel).all()
        assert instance._model.body("mug").name == "mug"
        assert instance._data.qpos.shape == first_qpos.shape
    finally:
        instance.shutdown()


def test_backend_starts_from_srdf_ready_state(backend: MuJoCoBackend) -> None:
    state = backend.get_robot_state()
    positions = dict(zip(state.name, state.position, strict=True))

    expected = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
    }
    for joint_name, joint_value in expected.items():
        assert positions[joint_name] == pytest.approx(joint_value, abs=1e-3)


def test_list_and_get_sensors(backend: MuJoCoBackend) -> None:
    sensors = {entry.name: entry.type for entry in backend.list_sensors().entries}

    assert sensors["joint_states"]
    assert sensors["world_camera"]
    assert sensors["ft_sensor_force"]
    assert sensors["ft_sensor_torque"]

    sensor_data = backend.get_sensors(
        ["joint_states", "world_camera", "ft_sensor_force", "ft_sensor_torque"]
    )
    assert len(sensor_data.joints) == 1
    assert sensor_data.joints[0].name == "joint_states"
    assert len(sensor_data.images) == 1
    assert sensor_data.images[0].width == 320
    assert sensor_data.images[0].height == 240
    assert len(sensor_data.forces) == 1
    assert len(sensor_data.torques) == 1


def test_camera_rendering_remains_valid_across_threads(backend: MuJoCoBackend) -> None:
    first_image = backend.get_sensors(["world_camera"]).images[0]

    with ThreadPoolExecutor(max_workers=1) as executor:
        second_image = executor.submit(
            lambda: backend.get_sensors(["world_camera"]).images[0]
        ).result()

    first_frame = np.frombuffer(first_image.data, dtype=np.uint8).reshape(
        first_image.height,
        first_image.width,
        3,
    )
    second_frame = np.frombuffer(second_image.data, dtype=np.uint8).reshape(
        second_image.height,
        second_image.width,
        3,
    )

    assert float(first_frame.mean()) > 20.0
    assert float(first_frame.std()) > 20.0
    assert float(second_frame.mean()) > 20.0
    assert float(second_frame.std()) > 20.0
    assert np.mean(
        np.abs(second_frame.astype(np.int16) - first_frame.astype(np.int16))
    ) < 10.0


def test_set_joint_target_and_reset_world(backend: MuJoCoBackend) -> None:
    initial = backend.get_robot_state()
    initial_finger = initial.position[7]

    backend.set_joint_target(
        names=["panda_finger_joint1", "panda_finger_joint2"],
        data=[0.03, 0.03],
        mode=core_pb2.JointCommand.ControlMode.POSITION,
        group="panda_hand",
    )

    moved = _wait_for_condition(
        lambda: backend.get_robot_state().position[7] > initial_finger + 1e-3,
        timeout=1.0,
    )
    assert moved

    backend.reset_world(seed=0, randomization_params={})

    reset = _wait_for_condition(
        lambda: abs(backend.get_robot_state().position[7]) < 1e-4,
        timeout=1.0,
    )
    assert reset


def test_joint_command_state_stays_replayable_during_velocity_control(
    backend: MuJoCoBackend,
) -> None:
    initial_command_state = backend.get_joint_command_state()
    initial_position_map = dict(
        zip(initial_command_state.name, initial_command_state.position, strict=True)
    )
    backend.set_joint_target(
        names=["panda_joint2"],
        data=[0.2],
        mode=core_pb2.JointCommand.ControlMode.VELOCITY,
        group="panda_arm",
    )
    time.sleep(0.05)

    robot_state = backend.get_robot_state()
    command_state = backend.get_joint_command_state()
    position_map = dict(zip(command_state.name, command_state.position, strict=True))
    velocity_map = dict(zip(command_state.name, command_state.velocity, strict=True))
    robot_position_map = dict(zip(robot_state.name, robot_state.position, strict=True))

    assert position_map["panda_joint2"] == pytest.approx(
        robot_position_map["panda_joint2"],
        abs=1e-3,
    )
    assert velocity_map["panda_joint2"] == pytest.approx(0.2)
    assert position_map["panda_joint6"] == pytest.approx(
        initial_position_map["panda_joint6"],
        abs=1e-6,
    )


def test_idle_loop_holds_initial_configuration(backend: MuJoCoBackend) -> None:
    initial = backend.get_robot_state().position[:7]
    time.sleep(0.3)
    later = backend.get_robot_state().position[:7]

    max_drift = max(
        abs(current - reference)
        for reference, current in zip(initial, later, strict=True)
    )
    assert max_drift < 0.01


def test_free_base_g1_idle_loop_holds_root_pose() -> None:
    backend = MuJoCoBackend(str(G1_29DOF_SCENE_PATH), headless=True)
    try:
        assert backend.robot_name == "g1_29dof"
        pelvis_id = mujoco.mj_name2id(
            backend._model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        initial_z = float(backend._data.xpos[pelvis_id][2])
        time.sleep(0.5)
        later_z = float(backend._data.xpos[pelvis_id][2])
    finally:
        backend.shutdown()

    assert later_z == pytest.approx(initial_z, abs=0.03)


def test_free_base_g1_twist_servo_does_not_flail_idle_joints() -> None:
    backend = MuJoCoBackend(str(G1_29DOF_SCENE_PATH), headless=True)
    try:
        command = core_pb2.ServoCommand(
            twist_cmd=core_pb2.TwistCommand(
                twist=common_pb2.TwistStamped(
                    twist=common_pb2.Twist(linear=common_pb2.Point(x=0.02))
                ),
                target_ee=core_pb2.EESpec(
                    name="left_wrist_yaw_link",
                    parent_jmg_name="left_arm",
                    group_name="left_arm",
                ),
            )
        )
        next(backend.servo_control_stream(iter([command])))
        time.sleep(0.2)

        state = backend.get_robot_state()
        velocities = dict(zip(state.name, state.velocity, strict=True))
    finally:
        backend.shutdown()

    idle_joint_names = [
        name
        for name in velocities
        if not name.startswith("left_shoulder")
        and not name.startswith("left_elbow")
        and not name.startswith("left_wrist")
    ]
    assert max(abs(velocities[name]) for name in idle_joint_names) < 1e-6
    assert max(abs(value) for value in velocities.values()) <= 1.05


def test_free_base_g1_spec_exposes_leg_end_effectors() -> None:
    backend = MuJoCoBackend(str(G1_29DOF_SCENE_PATH), headless=True)
    try:
        groups = {group.name: group for group in backend.get_robot_spec().joint_model_groups}
    finally:
        backend.shutdown()

    assert [ee.name for ee in groups["left_leg"].end_effectors] == [
        "left_ankle_roll_link"
    ]
    assert [ee.name for ee in groups["right_leg"].end_effectors] == [
        "right_ankle_roll_link"
    ]
    assert list(groups["both_legs"].joint_names) == [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]


def test_free_base_g1_leg_twist_servo_does_not_flail_other_joints() -> None:
    backend = MuJoCoBackend(str(G1_29DOF_SCENE_PATH), headless=True)
    try:
        pelvis_id = mujoco.mj_name2id(
            backend._model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        initial_z = float(backend._data.xpos[pelvis_id][2])
        command = core_pb2.ServoCommand(
            twist_cmd=core_pb2.TwistCommand(
                twist=common_pb2.TwistStamped(
                    twist=common_pb2.Twist(linear=common_pb2.Point(x=0.02))
                ),
                target_ee=core_pb2.EESpec(
                    name="left_ankle_roll_link",
                    parent_jmg_name="left_leg",
                    group_name="left_leg",
                ),
            )
        )
        next(backend.servo_control_stream(iter([command])))
        time.sleep(0.2)

        state = backend.get_robot_state()
        velocities = dict(zip(state.name, state.velocity, strict=True))
        later_z = float(backend._data.xpos[pelvis_id][2])
    finally:
        backend.shutdown()

    idle_joint_names = [name for name in velocities if not name.startswith("left_")]
    assert later_z == pytest.approx(initial_z, abs=0.03)
    assert max(abs(velocities[name]) for name in idle_joint_names) < 1e-6
    assert max(abs(value) for value in velocities.values()) <= 1.05


def test_servo_control_stream_accepts_twist(backend: MuJoCoBackend) -> None:
    command = core_pb2.ServoCommand(
        twist_cmd=core_pb2.TwistCommand(
            twist=common_pb2.TwistStamped(),
            target_ee=core_pb2.EESpec(name="hand", parent_jmg_name="panda_arm"),
        )
    )
    command.twist_cmd.twist.twist.linear.z = 0.02

    states = backend.servo_control_stream(iter([command]))
    state = next(states)
    assert len(state.name) == 9


def test_get_end_effector_state(backend: MuJoCoBackend) -> None:
    ee_state = backend.get_end_effector_state("panda_arm")

    assert ee_state.pose_stamped.header.frame_id == "world"
    orientation = ee_state.pose_stamped.pose.orientation
    assert any(
        abs(value) > 0.0
        for value in (orientation.x, orientation.y, orientation.z, orientation.w)
    )
