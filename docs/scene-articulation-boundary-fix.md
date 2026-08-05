# 场景铰接与机器人关节的边界修正

## 问题

我在对接时example场景的 MuJoCo CSD realization（4×4 房间 + 双开门柜 + 凳子 + Panda）暴露
了 `MuJoCoBackend` 的一个机器人边界判定缺陷。

`_collect_robot_body_ids` 我之前的实现是遍历所有根 body 子树，**只要子树内含 hinge 或
slide 关节就整棵并入机器人**（判据 `_is_robot_joint_type` 仅认 HINGE/SLIDE）。
柜子的 2 个抽屉滑轨（slide）与 2 扇门铰链（hinge）因此被认定为机器人关节。

实测后果（`check_provider_scene.py` 输出）：

```
OK  get_robot_state 返回 13 个可控关节:
     panda_joint1..7, panda_finger_joint1/2
  >> World_Objects_cabinet_double_door_01/PrismaticJoint_double_door_3_left_joint
  >> World_Objects_cabinet_double_door_01/PrismaticJoint_double_door_3_right_joint
  >> World_Objects_cabinet_double_door_01/RevoluteJoint_double_door_3_left_joint
  >> World_Objects_cabinet_double_door_01/RevoluteJoint_double_door_3_right_joint
```

两个具体问题：

1. `observation.state` 维度随场景家具数量漂移，与既有数据集不兼容；
2. 这些关节进入 idle hold 集合，每个物理步收到 `POSITION_KP=200 / KD=30` 的
   PD 保持力矩（backend.py:1087-1110），柜门与抽屉被"焊死"——**开抽屉、开柜门
   这类任务在当前实现下不可能成功**。

## 修复

机器人边界改以 **SRDF 声明**为准（新增 `_srdf_declared_body_ids`）：

- 收集 SRDF 中全部 `<joint>` / `<passive_joint>` 名与 `<link>` / `<chain>` 的
  link 名，映射回 MJCF body；
- 向上补齐到根的父链（末端 link 常不带关节），再向下取这些 body 的子树闭包，
  使运动学链完整；
- SRDF 缺失或名称对不上（例如机器人经 `attach` 带 prefix 引入）时，退回按
  机器人根子树界定，而不是原来的全场景扫描。

选择 SRDF 而非"单一根子树"的原因：**G1 双臂场景是真正的多根机器人**，左右臂各
自成根（各 7 个 hinge），单根假设会丢掉右臂——这在实现过程中由
`test_g1_dual_arm_spec_exposes_both_arms` 回归捕获。SRDF 是唯一能同时正确描述
"多根同属一个机器人"与"同根场景家具不属于机器人"的现有依据。

## 测试

新增 `tests/test_scene_articulation_isolation.py`（4 用例）+ 自包含 fixture
`tests/fixtures/articulated_scene/`（Panda + 带滑轨抽屉与铰链门的柜子；
`assets` 为指向 franka_panda 资产的符号链接，以满足 `meshdir="assets"` 相对
顶层 XML 解析的规则）：

- 状态向量只含 Panda 关节；
- 场景铰接自由度的 `qfrc_applied` 恒为 0（未被保持力矩驱动）；
- 抽屉被外部置位后能保持在该位置（未被拉回）；
- `RobotSpecification.joints` 只覆盖机器人关节。

修复前 3 个失败、1 个通过；修复后 4/4 通过。

## 验证

- 新测试 4/4；`test_g1_dual_arm_backend` 4/4、`test_mujoco_backend` 16/16；
- 全量回归 234 通过；
- ruff / mypy 全绿；
- **真实 example 场景实证**：关节数 13 → 9，柜门抽屉不再混入；同一场景的
  ExecuteJointTrajectory 末态误差不变（max 0.0135 rad / rms 0.0104 rad）。

## 附带新增

`check_provider_scene.py` — 场景包对接检查脚本（只读）。对任意提供方的
`scene.xml` 逐项验证：MuJoCo 原生加载与物理稳定性、vsim 后端推导出的关节集
（混入的场景关节以 `>>` 标出）、SRDF 组解析、逐相机渲染、
ExecuteJointTrajectory 实测误差、场景物体相对 Panda 工作半径 0.855m 的可达性。

用法：

```bash
cd vsim && MUJOCO_GL=egl PYTHONPATH=$PWD python check_provider_scene.py <scene.xml>
```

## 已知边界

- 机器人经 `attach` + prefix 引入时，SRDF 里的 link/joint 名不带 prefix，
  `_srdf_declared_body_ids` 会返回空并退回根子树启发式。当前对接的场景用
  `include` 引入机器人（无 prefix），不受影响；若将来提供场景时改用 attach 引入
  机器人，需要补 prefix 映射。
- `_find_robot_root_body` 未改动：多根机器人下它仍只选关节最多的那个根，用于
  `frame_id` 与 root hold；这是既有行为，本次不在范围内。
