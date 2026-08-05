"""Check whether a provider-supplied MuJoCo CSD realization works with vsim.

Read-only: loads the scene through MuJoCoBackend, inspects the joint set the
runtime derives from it, renders every camera, runs one short
ExecuteJointTrajectory, and reports reachability of scene objects against the
Panda workspace. Nothing under the scene directory is modified.

Usage (from the vsim directory):
    MUJOCO_GL=egl PYTHONPATH=$PWD python check_provider_scene.py /path/to/scene.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from robosim.backends import MuJoCoBackend

ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
PANDA_REACH_M = 0.855  # official max reach, frankaemika.github.io
COLLECTION_CAMERAS = ("world_camera", "wrist_camera")


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_native(scene: Path) -> mujoco.MjModel:
    rule("1. MuJoCo 原生加载")
    model = mujoco.MjModel.from_xml_path(str(scene))
    print(f"  OK  nbody={model.nbody} nq={model.nq} nv={model.nv} "
          f"nu={model.nu} ngeom={model.ngeom} ncam={model.ncam}")
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    finite = bool(np.isfinite(data.qpos).all())
    print(f"  200 步物理: t={data.time:.3f}s 有限={finite} "
          f"max|qvel|={np.abs(data.qvel).max():.4f}")
    cameras = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
               for i in range(model.ncam)]
    print(f"  相机: {cameras}")
    missing = [name for name in COLLECTION_CAMERAS if name not in cameras]
    if missing:
        print(f"  [差异] 采集管线需要的相机缺失: {missing}")
    return model


def check_backend(scene: Path) -> None:
    rule("2. vsim MuJoCoBackend 加载 + 运行时推导出的关节集")
    backend = MuJoCoBackend(str(scene), headless=True)
    try:
        state = backend.get_robot_state()
        print(f"  OK  get_robot_state 返回 {len(state.name)} 个可控关节:")
        expected = set(ARM_JOINTS + FINGER_JOINTS)
        strays = []
        for name, position in zip(state.name, state.position, strict=True):
            mark = "   " if name in expected else ">> "
            if name not in expected:
                strays.append(name)
            print(f"    {mark}{name:52s} {position:+.4f}")
        if strays:
            print(f"\n  [重要] {len(strays)} 个非机器人关节被当成机器人关节纳入状态向量:")
            for name in strays:
                print(f"        {name}")
            print("        后果: 采集到的 observation.state 维度与我方数据集不一致，")
            print("        且这些关节每步会被施加 PD 保持力矩（柜门/抽屉会被'焊住'）。")
        else:
            print("\n  关节集干净: 仅含 Panda 自身关节。")

        spec = backend.get_robot_spec()
        print(f"\n  robot_name={spec.robot_name}")
        for group in spec.joint_model_groups:
            print(f"    group {group.name:16s} joints={list(group.joint_names)}")

        rule("3. 渲染每个相机")
        model = mujoco.MjModel.from_xml_path(str(scene))
        for index in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            try:
                sensors = backend.get_sensors([name])
                image = sensors.images[0]
                array = np.frombuffer(image.data, dtype=np.uint8).reshape(
                    image.height, image.width, 3
                )
                out = Path(f"/tmp/provider_scene_{name}.png")
                try:
                    from PIL import Image

                    Image.fromarray(array).save(out)
                    saved = f" -> {out}"
                except ImportError:
                    saved = ""
                print(f"  OK  {name}: {image.width}x{image.height} "
                      f"均值={array.mean():.1f} 标准差={array.std():.1f}{saved}")
            except Exception as error:  # noqa: BLE001 - report, don't abort
                print(f"  FAIL {name}: {type(error).__name__}: {error}")

        rule("4. ExecuteJointTrajectory 实测（2.5s @30fps）")
        target = HOME_Q + np.array([0.20, 0.18, 0.10, 0.15, 0.15, -0.17, 0.11])
        points = []
        frames = 75
        for frame in range(1, frames + 1):
            alpha = frame / frames
            q = HOME_Q + (target - HOME_Q) * alpha
            points.append(((frame) / 30.0, [float(v) for v in q] + [0.03, 0.03]))
        try:
            backend.execute_joint_trajectory(
                ARM_JOINTS + FINGER_JOINTS, points, "panda_arm_hand"
            )
            state = backend.get_robot_state()
            actual = {n: p for n, p in zip(state.name, state.position, strict=True)}
            errors = np.array([abs(actual[n] - t)
                               for n, t in zip(ARM_JOINTS, target, strict=True)])
            print(f"  OK  轨迹执行完成  末态误差 max={errors.max():.4f}rad "
                  f"rms={np.sqrt((errors ** 2).mean()):.4f}rad")
            for name, t, err in zip(ARM_JOINTS, target, errors, strict=True):
                print(f"    {name:16s} 目标={t:+.4f} 实际={actual[name]:+.4f} 误差={err:.4f}")
        except Exception as error:  # noqa: BLE001 - report, don't abort
            print(f"  FAIL {type(error).__name__}: {error}")

        rule("5. 场景物体可达性（Panda 最大触及 0.855m）")
        model_b = mujoco.MjModel.from_xml_path(str(scene))
        data_b = mujoco.MjData(model_b)
        mujoco.mj_forward(model_b, data_b)
        base_id = mujoco.mj_name2id(model_b, mujoco.mjtObj.mjOBJ_BODY, "link0")
        base = data_b.xpos[base_id] if base_id >= 0 else np.zeros(3)
        print(f"  机器人基座 link0 @ {np.round(base, 3).tolist()}")
        for body_id in range(1, model_b.nbody):
            name = mujoco.mj_id2name(model_b, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None or name.startswith(("link", "hand", "left_finger",
                                                "right_finger", "world")):
                continue
            position = data_b.xpos[body_id]
            distance = float(np.linalg.norm(position[:2] - base[:2]))
            verdict = "可达" if distance <= PANDA_REACH_M else "超出工作半径"
            print(f"    {name:56s} 水平距离={distance:.3f}m  {verdict}")
    finally:
        backend.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="path to the provider's scene.xml")
    args = parser.parse_args(argv)
    check_native(args.scene)
    check_backend(args.scene)
    rule("完成：以上为只读检查，未修改被测场景任何文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
