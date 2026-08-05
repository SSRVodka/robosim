# 轨迹合成与跨后端重放：本轮变更记录

由我survey的轨迹合成与跨后端采集第一节管线与第六节验收标准的实施。本文是本 PR 全部改动的记录。

## 新增

- `control_stubs/tools/traj_synthesis.py` — 合成库：物体坐标系 Keypose 模板
  （与 scripted_collect 同语义的 pick-place 六段）→ ruckig 笛卡尔逐段
  state-to-state 时间参数化 → mink 微分 IK（FrameTask(hand) + PostureTask 冗
  余锚定，非臂 DOF 速度置零，显式收敛检查）→ 30fps `(t, q[7], gripper)` 指令
  流 → 校验门（关节限位余量、逐帧速度 vs 官方 Panda 限值×0.8、IK 残差、相位
  感知运动学接触扫描：合爪前禁止机器人∩{cup,table,container}，合爪后禁止
  ∩{table,container}）。失败抛 `SynthesisError`，携带 `synthesis_blocker/v1`
  typed payload。指令流工件为 JSONL：首行 meta（fps/seed/物体位姿/IK 最差残
  差/库版本/scene hash），逐行样本。
- `control_stubs/tools/synth_collect.py` — 拒绝采样采集 CLI：随机杯位 → 合成
  +门检（失败落 blocker 不碰仿真器）→ episode_start → ExecuteJointTrajectory
  → 成功谓词（GetObjectPose 连续 10 次轮询杯在容器界内，LIBERO 式）→ 成功
  episode_end / 失败 episode_cancel + blocker。manifest.jsonl 记质量元数据
  （seed、杯位、时长、IK 残差、谓词证据）。
- `control_stubs/tools/replay_stream_pybullet.py` — 跨后端重放：暂停后端墙钟
  循环，按 meta.object_positions 复位场景初态后，每帧 set_joint_target +
  手动 step_physics×8（30fps 与 1/240 步长整除，确定性、与墙钟无关），后端
  本地谓词独立判成功，输出含臂部跟踪误差的 JSON 证据。
- `tests/fixtures/pybullet_cup_scene/{scene.py,scene_meta.json}` — PyBullet
  杯子场景 fixture（几何对齐 MJCF；有损转换显式记录于 meta：杯把手省略、接
  触参数按后端调优不做数值转换）。
- `tests/test_traj_synthesis.py`（7 用例，并且有 ruckig 云 API 单测守卫）、
  `tests/test_object_pose_rpc.py`（2）、`tests/test_pybullet_replay.py`（1）。

## 修改

- `control_stubs/control_stubs/simulation.proto` — SimulationService 新增
  `GetObjectPose(ObjectPoseRequest) returns (ObjectPoseReply)`（评估器读物体
  位姿），并再生 simulation 的 Python stub。
- `robosim/backends/mujoco/backend.py`、`robosim/backends/pybullet/backend.py`
  — 各加 `get_object_pose`（镜像既有 set；MuJoCo 读 body xpos/xquat 并转
  xyzw，PyBullet 走 getBasePositionAndOrientation）。
- `robosim/grpc_server/simulation.py` — GetObjectPose servicer（沿用既有
  UNIMPLEMENTED/INTERNAL 错误模式）。
- `control_stubs/tools/client.py` — `SimulationStub.get_object_pose`。

## 依赖

pip 新增 mink 1.2.0 / ruckig 0.19.4 / scipy 1.18.0（+qpsolvers、daqp）；
dry-run 与安装后均验证 numpy 2.2.6、grpcio 1.78.1、protobuf 6.33.5 未动。
这个环境的栈原本就由 pip 管理（conda 那边没有 numpy/scipy），之前我survey的方案中先 mamba 装
scipy的措施不太合适，直接 pip 一致安装。

## 相对我survey的文档的实现细化

1. ruckig 在**笛卡尔空间**逐段参数化、mink 对每个 30fps 采样点解 IK（survey
   原本是"IK 关键位姿 + ruckig 关节段"）：保证抓取下降段手部严格直线，避免
   关节空间插值把手部路径弯进杯沿；ruckig 仍只用 state-to-state 单段（云
   API 红线不变）。
2. 重放端从指令流 meta 复位物体初态：我一开始遗漏了导致 PyBullet 重放 1/3 成功
   （夹爪在错误杯位下抓），修正后 3/3——教训已写入survey文档的可能的问题语义
   （指令流仅对其合成时的物体布局有效）。

## 验收证据（对照survey的文档第六节）

1. 合成端到端：正式批量 50 集 **50 攻 50 成（100%，零 blocker）**，约 16s/集 
    ，manifest 含全套质量元数据；门负路径三条（超速/不可
   达/穿透）单测拦截；
2. 训 ACT 对比基线：合成 50 集（seed 200–249，±5cm 随机杯位）训 ACT 50k
   （final loss 0.032，配置与手写基线逐字一致），六杯位评估 **5/6，优于手写
   基线 4/6**——分布内 5/5（含基线搬运脱落的 (+5,+5) 角点），仅分布外
   +7cm 失败（两者皆败）。相位紧跟合成专家（抓取 ~3.1–4.0s，随杯位变化）；
3. 跨后端：三条 MuJoCo 指令流在 PyBullet 全部重放成功（臂部跟踪误差
   ≤0.026 rad），杯子静止终态两引擎系统性不同（MuJoCo ~[0.39,0.30] vs
   PyBullet ~[0.29-0.39,0.37-0.40]）——帧级发散、任务级语义等价的直接证据；
4. 回归：227 通过；ruff/mypy 全绿。

## 已知边界

- MuJoCo 采集路径仍走服务端 ExecuteJointTrajectory（sim-time 配速），
  PyBullet 重放为进程内手动步进——Gazebo 后端接入同一指令流等我跑通环境；
- 谓词的容器界为常量（±0.15m、z>0.05），后续应随 CSD 评估器定义生成；
- PyBullet 场景为手写 fixture，正式路径应由 CSD 编译器产出（本 fixture 兼
  作其目标形态参考）。
