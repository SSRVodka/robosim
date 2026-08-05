# 变更记录：scripted expert 数据采集 MVP

本文档记录一次独立迭代的新增内容。

## 新增内容

### 1. `control_stubs/tools/scripted_collect.py`（新客户端工具）

脚本化 pick_and_place 专家采集器。在现有 Franka Panda MuJoCo 场景
（红色长方体 + 绿色容器 + world/wrist 相机）上自动执行
reach → grasp → lift → place，并通过 `RobotDataService` 录制 LeRobotDataset。

设计要点：

- **控制全部走 vsim gRPC**（`RobosimClient`），不 bypass 框架。MuJoCo bindings
  仅在客户端作运动学库（`PandaKinematics`，damped-least-squares IK，位置 +
  竖直向下姿态，4~8 次迭代收敛）；
- 专家输出为**关节位置目标流** `(arm_q[7], gripper)`，按 `--control-fps` 下发到
  `panda_arm` / `panda_hand` 两个 JMG。关节流与后端解耦——后续跨后端同步采集
  只需把同一目标流广播到多个 server 实例，无需在其他后端重解 IK；
- 支持 `--randomize-box`：通过 `SetObjectPose` 随机化 box 初始位置（±5cm，
  seed 可复现），抓取目标随实际位置更新；
- `--exclude-sensors`：录制侧传感器排除（透传 `RecordOptions.sensor_name_excluded`）。

### 2. `MuJoCoBackend.set_object_pose`（新后端方法）

MuJoCo 后端此前未实现 `SetObjectPose` RPC（仅 PyBullet 有）。新增最小实现：
写 freejoint qpos（proto 四元数 xyzw → MuJoCo wxyz），清零 qvel，`mj_forward`。
非 freejoint 物体报 `ValueError`。

### 3. `tests/test_scripted_collect.py`

IK 收敛（<1mm）、不可达目标报错、目标流有限性/连续性、终点在容器上方且开爪。

## 验证证据

- 单测：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_scripted_collect.py -q`
  → 4 passed；`ruff check` / `mypy` 通过；
- 端到端：`python -m robosim.server --backend mujoco --scene <panda scene.xml>
  --port 50061 --headless` + 采集 CLI，共采集 30 episodes（2851 帧 @5fps），
  **逐集末帧图像抽查 30/30 box 落入容器**（100% 成功率）；
- 数据可被 `LeRobotDataset` 正常加载，含 `observation.state/velocity/effort`、
  `action`（关节位置目标）、EE 位姿、`observation.images.world_camera`。

## 采集效率（本机 WSL2，RTX 4070 Laptop）

| 配置 | 单集耗时 | 10 集 | 说明 |
| --- | --- | --- | --- |
| 20Hz 控制 + 双相机 5fps 录制 | ~9.0s | 90.2s | 抓取失败（见 blocker 1） |
| 10Hz 控制 + 单相机 5fps 录制（最终配置） | 20~23s | ~211s | 30/30 成功 |

100 集推算约 35~40 分钟（含每 ~10 集重启 server 的平台开销，见 blocker 3）。

## Blockers / 已知限制（记录于本文档，学长可查看）

1. **recorder 渲染阻塞物理步进**（vsim runtime，应该是属于Part D 范围）：
   `_render_camera_locked` 持 `_state_lock` 渲染（WSL 下 66~137ms/帧），录制期间
   物理步进线程被周期性阻塞，仿真时间显著慢于墙钟。按墙钟下发目标流的控制器
   会因此严重滞后——这是 20Hz 采集配置抓取失败的根因。建议渲染使用状态快照，
   物理锁外执行。
2. **recorder fps 静默降速**：请求的录制 fps 超出渲染能力时，采样循环实际帧率
   下降但数据集仍标称请求值（首次实测：请求 20fps 实录 ~3.2fps），时间戳失真会
   影响训练/回放。建议超限时告警或写入实测 fps。
3. **WSL2 Mesa D3D12 渲染堆崩溃**：连续渲染数千帧后 server 进程
   `corrupted double-linked list` SIGABRT（EGL 与默认 GL 均复现）。我的WSL2问题，
   实验室用原生 Linux 应该不存在。临时规程：每 ~10 集一批，批间重启 server；
   LeRobotDataset 按集落盘 + resume，崩溃不丢已保存数据。
4. **wrist_camera 默认排除**：为把渲染负载降到物理可承受范围（blocker 1 的
   规避），本机采集只录 world_camera。正式采集环境应恢复双相机。
5. `--control-fps` 隐含假设 PD 跟踪能力（KP=200 偏软）：抓取/释放前有
   `SETTLE_FRAMES` 收敛等待。若后端增益调整，这些帧数可减小。

## 使用示例

```bash
# 起 server（headless）
python -m robosim.server --backend mujoco \
  --scene drivers_sim/mujoco/assets/robots/franka_panda/scene.xml \
  --port 50061 --headless

# 采 10 集（随机化 box，10Hz 控制，5fps 录制，仅 world 相机）
python -m control_stubs.tools.scripted_collect --port 50061 \
  --scene drivers_sim/mujoco/assets/robots/franka_panda/scene.xml \
  --repo-name pick_place_v3 --episodes 10 --control-fps 10 \
  --exclude-sensors wrist_camera --randomize-box --seed 0
```
