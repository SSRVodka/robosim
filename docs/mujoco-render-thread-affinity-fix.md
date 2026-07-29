# MuJoCo 离屏渲染线程亲和性修复

## 现象

`pick_place_cup_v2` 采集（50 集计划）在第 13 集后 server 进程 abort：

- 客户端：`grpc UNAVAILABLE ... Connection refused`（端口 50061）；
- server 日志末行：`corrupted double-linked list`（glibc 堆损坏 abort）。

## 根因

`_render_camera` 以 `(threading.get_ident(), width, height)` 作为
`mujoco.Renderer` 缓存 key，意图是"每线程一个 renderer"。但：

1. `LerobotDataRecorder.episode_start` **每个 episode 新建一个采样线程**；
2. glibc 会复用已退出线程的 ident——实测顺序创建 20 个线程，20 个全部拿到
   同一个 ident（unique: 1 of 20）；
3. `MUJOCO_GL` 未设置时 renderer 使用 GLFW 后端（`mujoco/glfw/__init__.py`），
   GL context 与创建它的线程绑定（thread-affine）。

三者叠加：第 N+1 集的新线程命中第 N 集已死线程创建的 renderer 缓存，跨线程
使用其 GL context → 未定义行为 → 堆损坏。崩溃门槛取决于堆布局，非确定性
（v1 曾 11 集通过，v2 第 13 集崩溃）。

同一缺陷还有两处伴生问题：

- 每集泄漏一对 renderer（缓存 key 里的死 ident 永不再释放，直到 shutdown）；
- `shutdown()` 在调用者线程 `close()` 所有 renderer——同样是跨线程释放
  GL context 的未定义行为；
- `get_sensors`（gRPC / policy 推理路径）的 `_renderers` 缓存有同样的
  ident-key 问题（gRPC 线程池线程同样会轮换）。

## 修复设计

所有离屏渲染收敛到 backend 内**唯一常驻渲染线程**（`mujoco_backend_render`）：

- renderer 的创建、复用、释放全部发生在该线程上，满足 GL 线程亲和性；
- 缓存 key 只剩分辨率 `(width, height)`，泄漏随线程无关化消失；
- 任意线程（gRPC handler、采样消费线程）通过 `_RenderRequest` 队列提交渲染，
  `threading.Event` 等待结果；渲染异常经 `request.error` 回传原线程抛出；
- `shutdown()` 投递 `None` 哨兵，渲染线程在退出前于本线程释放全部 renderer；
  等待方在 `_stop_event` 置位后放弃等待，避免关停竞态挂死；
- 删除 `_renderers` / `_capture_renderers` 两个 ident-key 字典及
  `_read_sensor_data` 的 renderers 参数。

录制与 policy 互斥（ActivityCoordinator），两路渲染不会同时高频出现，单线程
串行化没有实际并发损失。

## 验证

- 新增回归测试 `test_render_survives_sequential_consumer_threads`：3 个顺序
  消费线程各带相机采样（即崩溃配方），断言图像完整且 gRPC 路径同样可渲染；
- `tests/test_sim_capture.py` + `tests/test_joint_trajectory.py` 9/9 通过；
  全量 `pytest tests/` 217 通过、8 失败——8 个失败均为 OpenUSD 用例的环境
  预存问题（`usdchecker` PermissionError），已在未含本修复的树上复现同样
  失败，与本改动无关；
- `ruff` / `mypy` 通过；
- 端到端：重启 server 后从第 13 集续采 `pick_place_cup_v2` 37 集
  （seed 13–49），**一次性全部完成、server 全程存活**：
  `collected 37 episodes in 3633.4s (mean 98.2s)`。

## 采集结果（pick_place_cup_v2 最终状态）

- 50 episodes / 24890 frames / 30 fps（`--randomize-object`，杯子初始 XY
  ±5 cm，seed 0–49，任务文本与场景其余部分固定）；
- 逐集校验：帧长 464–529；相邻帧 timestamp 步长全部为 1/30 仿真秒；
- 首帧图像随 seed 两两存在真实差异（杯子位置像素级变化，max_abs_diff=1.0），
  随机化确认写入数据；各集末帧一致符合预期（末态相同：杯子放入同一容器位、
  机械臂同一收尾姿态）。

## 对 blocker #4（WSL2 稳定性）结论的更正

之前文档以"11 集 / 5467 帧连续采集成功"得出 WSL2 崩溃已消失的结论，**该结论
不成立**：本次第 13 集的堆损坏证明当时只是未达触发条件。且根因并非
WSL2 / Mesa D3D12 驱动，而是上述跨线程 GL 复用缺陷（我自己的代码的原因），
在任何平台都可能触发。修复后 WSL2 默认 GLFW 后端连续 37 集零崩溃（含此前
13 集，同一数据集共 50 集）。早前迭代把渲染不稳定归因于 WSL2 环境的判断
需要重新审视；就当前采集工作负载而言，修复后的本机 WSL2 环境是稳定的。
