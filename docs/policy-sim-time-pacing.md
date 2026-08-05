# Policy 推理循环仿真时间对齐

## 背景与量化证据

`LerobotPolicyRunner._run_loop` 原实现按**墙钟** 1/control_fps 定节奏，且观测经
`get_sensors` 在 `_state_lock` 内等待渲染。用 ACT 10k checkpoint（CPU 推理）
在 MuJoCo 上实测 60s：

- 仿真时间仅前进 9.78s（16% 实时），观测在仿真时间轴上的间隔 p50=24ms /
  p90=96ms / max=114ms——训练数据是严格 33.3ms；平均值接近只是 CPU 推理耗时
  恰好与物理实时速率相当的巧合，换设备（GPU 推理）就有可能失效了；
- 行为后果：臂部相位与专家相关系数 0.89–0.92，抓取窗口内夹爪从未闭合，
  任务失败。

这与录制路径修复前的缺陷同类（数据/控制时间轴被墙钟与渲染耗时扭曲），违反了
这个 PR 之前确立的数据时间轴 = 仿真时间的原则。

## 设计

推理观测复用与录制相同的 sim-capture 机制，保持一条时间轴语义：

- backend 支持采样挂点时（`getattr(backend, "start_sim_capture", None)`，与
  recorder 相同的探测方式），`_run_loop` 走 `_sim_capture_loop`：
  `start_sim_capture(control_fps)` → 循环 `next_sim_capture([], sensor_names)`
  → `observation_from_snapshot` → 推理 → `set_joint_target`；
- 观测间隔由捕获机制保证为**严格 1/control_fps 仿真秒**；推理慢时背压暂停
  仿真而不是跳帧或压缩时间轴；渲染发生在消费线程（经渲染线程），不再持
  `_state_lock` 阻塞物理；
- 动作延迟上界为缓冲池深度（SIM_CAPTURE_BUFFER_COUNT=4 个间隔 ≈133ms 仿真
  时间），仅在 chunk 边界前向耗时较长时短暂出现；
- 无挂点后端（Gazebo/PyBullet）保留原墙钟路径 `_wall_clock_loop`；
- `stop_policy` 在 join 前调用 `stop_sim_capture` 解除暂停态仿真下的消费阻塞
  （幂等，capture 未激活时为空操作）；
- `LerobotObservationAdapter` 抽出 `_build_observation` 共用体，新增
  `observation_from_snapshot(CaptureSnapshot)` 与 `sensor_names` 属性；
  墙钟路径的 `capture_observation` 语义不变。

## 测试调整

- 新增 `test_lerobot_policy_runner_prefers_sim_capture_loop`：带采样挂点的桩
  backend，断言 capture fps 传递、观测由采样流驱动、stop 正常终止；
- `test_lerobot_act_policy_runs_multistep_headless_on_franka_backend` 原以 spy
  `get_robot_state` 统计观测次数——新路径观测来自 snapshot 不再经过该方法，
  改为仅以 `set_joint_target` 命令数作为多步依据（MuJoCo 参数化用例现在实际
  运行的就是 sim-capture 路径）；
- 全部 policy 测试 12/12、采样/录制回归 11/11、ruff、mypy 通过。

## 修复后端到端证据（ACT 10k checkpoint，CPU 推理，MuJoCo）

- 观测节奏：由机制保证 33.3ms 仿真间隔；60s 墙钟推进 8.3 仿真秒是背压之下
  的预期（每 tick 等待双相机渲染 + 推理 ~240ms 墙钟），时间轴不再失真；
- 完整 episode（19.5 仿真秒）：7 关节全程与专家相关系数 0.987–0.998；抓取
  +7.07s（专家 +7.0s）、释放 +16.53s（专家 +16.5s）；末帧杯子在容器内；
- 泛化（杯子挪到训练未见的偏移位）：(+3,-3)cm 与 (-4,+2)cm 均成功抓放入容
  器，抓握保持段分别为 +6.91→17.71s 与 +6.33→15.06s；抓取时刻随杯位变化
  （6.33/6.91/7.07s），证明行为是视觉条件化的，不是轨迹背诵；
- 对照：同一 checkpoint 在修复前的墙钟路径下抓取失败（见上文量化证据）。

## 遗留说明

- 墙钟路径（无挂点后端）仍存在与修复前相同的节奏风险，属于对应后端接入
  采样挂点前的已知限制；
- 推理墙钟速率由渲染吞吐主导（WSL2 双相机 ~240ms/tick）；正确性优先，与
  录制路径的取舍一致。
