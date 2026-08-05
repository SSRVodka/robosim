# 变更记录：采集链路 trigger 化 + 30fps 仿真时间对齐录制

本文档记录针对 `feat/scripted-collection` 的修复迭代（4 个 commit），
上一迭代记录见 `docs/scripted-collection.md`。

## 问题与修复

| # | 问题                                | 修复                                                   | commit |
| --- |-----------------------------------|------------------------------------------------------| --- |
| 1 | 录制 5fps 太低，VLA 数据至少 30fps         | 录制改按**仿真时间**每 1/fps 秒严格采一帧，默认 30fps                  | 1 |
| 2 | 红长方体换成真实物品（杯子）、场景固定               | scene.xml 换瓷白马克杯（圆柱杯身+把手），固定初始位                      | 3 |
| 3 | 打通旧文档 blockers（渲染持锁、fps 静默降速）     | 渲染移出物理锁 + 背压保帧（见下）                                   | 1 |
| 4 | WSL2 Mesa 崩溃                      | 单线程渲染 + renderer 常驻复用；三 GL 后端实测（见下）                  | 1/4 |
| 5 | rollout 不走逐步 gRPC，gRPC 只做 trigger | 新增 `ExecuteJointTrajectory`：整条轨迹一次上传，server 内按仿真时间执行 | 2 |

MuJoCo backend 从未实现或者使用过 `StepPhysics`（servicer 对 MuJoCo 落
UNIMPLEMENTED，采集客户端也不调它），旧架构的真正瓶颈是
渲染在 `_state_lock` 内阻塞物理步进 + recorder 按墙钟采样。

## 设计变更

### 1. 仿真时间对齐录制（`MuJoCoBackend` + `LerobotDataRecorder`）

主要核心是**数据时间轴 = 仿真时间，与墙钟解耦**。

- backend 新增 `start_sim_capture(fps)` / `next_sim_capture(...)` /
  `stop_sim_capture()`：物理线程每当 `data.time` 跨过绝对采样时刻
  `k/fps`，把 MjData `mj_copyData` 进预分配缓冲池（4 个，<1ms）并入队；
  **池空时物理线程在锁外阻塞等待（背压）**——仿真降速而不是丢帧或降 fps；
- 消费线程（recorder 采样线程）出队后在**物理锁外**从副本提取状态并渲染
  （`Renderer.update_scene` 纯 CPU、`render()` 的 GL 全在该线程；renderer
  常驻复用，单线程 GL）；
- 采样时刻绝对累加（`next_time += 1/fps`），帧间抖动 ≤1 个物理步
  （2ms）且无漂移；帧时间戳为仿真时间；
- `LerobotDataRecorder` 探测 backend 采集挂点：支持则走仿真时间路径，
  否则（Gazebo/PyBullet）保留原墙钟路径，行为不变。

之前几条blocker状态：

1. 渲染持锁阻塞物理 → **已解**（锁内只剩 `mj_copyData`）；
2. fps 静默降速/时间戳失真 → **已解**（背压保帧，数据集标称 fps 即真实仿真采样率）；
3. WSL Mesa 崩溃 → 见下文实测；
4. wrist_camera 默认排除 → **已恢复**（默认双相机录制）；
5. PD 收敛等待 → dwell 改为秒语义（`SETTLE_SEC`/`GRIPPER_DWELL_SEC`），随
   control fps 自动换算。

### 2. rollout trigger 化（`ExecuteJointTrajectory`）

- `robot_core.proto` 新增
  `rpc ExecuteJointTrajectory(JointTrajectory) returns (Status)`：
  `points[]{positions[], time_from_start}`，`time_from_start` 为仿真秒，
  零阶保持位置目标，执行完（末点仿真时刻已过）才返回；
- MuJoCo backend 在物理循环内推进轨迹，每步按 `data.time - t0` 查表写
  `_control_targets`；Pause 时仿真时间不推进，轨迹本来就暂停；
  `emergency_stop` 中止轨迹；其他 backend 未实现时 RPC 返回 UNIMPLEMENTED；
- `scripted_collect.py`：每 episode 的 gRPC 调用从 ~2000 次（逐步下发）降到
  5 次（reset / set_object_pose(可选) / episode_start / execute_trajectory /
  episode_end）——gRPC 只是 trigger，rollout 全在 server 进程内；
- 专家轨迹加密：`CARTESIAN_STEP` 0.01→0.002 + control fps 10→50（EE 速度不变
  0.1m/s），30fps 采样帧间 action 均有变化。

### 3. 场景（`scene.xml`）（我没提交）

- `box`（红长方体）→ `cup`：圆柱杯身 r=0.028 h=0.10（直径 5.6cm < 夹爪开口
  8cm）+ 把手，瓷白色，质量 0.14kg，摩擦 2.2；固定初始位 (0.6, -0.2)；
- `--randomize-box` 更名 `--randomize-object`，默认关闭（固定场景）；
- 任务文本 "pick the cup and place it into the container"。

## WSL2 渲染后端实测（本机 RTX 4070 Laptop，320×240 双相机，新采集链路）

| MUJOCO_GL | 每帧耗时 | 说明 |
| --- | --- | --- |
| 默认（GLX/D3D12） | 90 ms | 最快 |
| egl（D3D12） | 107 ms | |
| osmesa（CPU 软渲染） | 256 ms | 绕开 D3D12 驱动的候选 |

双相机 30fps 需 ≤33ms/帧，WSL2 均不达实时 → 背压使仿真降速至 ~0.3×实时。
**数据不受影响**（严格 30fps 仿真时间轴），只影响墙钟吞吐。

稳定性与采集环境结论：

- 默认 GL 连续采集 **11 集 / 5467 帧 / 双相机 / 约 10 分钟连续渲染无崩溃**
  ，旧架构下数千帧后必现 `corrupted double-linked list` SIGABRT。判断该崩溃
  源于旧架构的多线程 GL 使用，渲染在 gRPC 线程池的任意线程上创建/使用
  renderer，新架构渲染全部固定在单一录制线程且 renderer 常驻复用后消失。

## 端到端验证

- 试采 1 集：497 帧 @30fps，双相机齐全，墙钟 57.6s；
- 批量 `pick_place_cup_v1`：**10 集 / 4970 帧 @30fps**，墙钟 543s（mean 54.3s/集，
  ≈0.31×实时，纯渲染受限）；
- **时间轴校验**：每集 497 帧、时间跨度 16.533s，恰好等于 (497−1)/30，
  10/10 集完全一致——帧间隔严格 1/30 仿真秒、零累积漂移，旧架构的 fps 失真
  彻底消除；
- 抓放成功：确定性轨迹下杯体从 (0.600, −0.200, 0.149) 移动到
  (0.395, 0.298, 0.119)，落在容器（中心 (0.4, 0.3)、内壁半宽 0.175）内，
  10/10 集一致（固定场景 + 同一轨迹）。

## 单测

- `tests/test_sim_capture.py`：采样间隔 = 1/fps ±1 物理步且无累积漂移；
  背压时仿真时间停滞不丢帧；stop 解除阻塞、仿真恢复；
- `tests/test_joint_trajectory.py`：轨迹按仿真时间执行到位、非法输入拒绝、
  pause 暂停执行、并发轨迹拒绝、action 命令状态可见；
- `tests/test_scripted_collect.py`：适配杯子场景与新签名。

## 使用示例

```bash
python -m robosim.server --backend mujoco \
  --scene drivers_sim/mujoco/assets/robots/franka_panda/scene.xml \
  --port 50061 --headless

python -m control_stubs.tools.scripted_collect --port 50061 \
  --scene drivers_sim/mujoco/assets/robots/franka_panda/scene.xml \
  --repo-name pick_place_cup_v1 --episodes 10 --seed 0
# 默认：50Hz 轨迹、30fps 录制、双相机、固定杯位
```
