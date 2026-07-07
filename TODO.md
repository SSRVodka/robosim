# TODO - 短期里程碑

# VERSION v0.0.5

## New Features & Details
- [x] 支持数据集（基于给定的 repo、episode）重放；
- [ ] (**<u>WIP</u>**) 基于 IL Policy（如 ACT）的推理支持（Action Chunking，或许可以借助 lerobot 的能力）；
- [ ] 基于 RL Policy 的推理支持；
- [ ] 支持常见模型的 VLA 推理（对接 LeRobot 接口）；
- [ ] 定义并实现 CSD realization 接口：接收 thesis-level benchmark
  generator 输出的 Concrete Scenario Definition，将其 finalize/load 为
  MuJoCo、Gazebo 等后端 native scene/artifacts，并返回可审计的 backend
  manifest；实现 MuJoCo 路径前必须先阅读官方 MJCF 语义文档；
- [ ] 为 CSD realization 定义缓存 key：CSD content hash、asset variant hash、
  backend target、realization config、`vsim` realization version、simulator
  version、sampled randomization values；
- [ ] 为 CSD realization 定义 asset backend compatibility 检查：mesh format、
  material/texture、collision、joint/articulation、sensor、lighting、scale、
  frame/up-axis、contact/inertial semantics；不支持或有损转换必须返回
  validation failure/blocker；
- [ ] 在实现 MuJoCo/Gazebo realization 前记录官方文档依据：MuJoCo MJCF XML
  Reference、ROS URDF XML documentation、SDFormat specification；

## Bug Fixes

- [x] 修复 MuJoCo recorder 在 velocity/twist servo 下把不可 replay 的原始 joint command 写入 `action`，导致 EpisodeReplay 将关节拉回 0；
- [ ] （优先级低）像 EpisodeReplay 这样的 gRPC 接口报错无法传递详细的错误信息，只能从服务端的 console 看到。比如故意输入不存在的 repoName，服务端报错正常，但是客户端只能收到模糊的错误信息 "UNKNOWN:Error received from peer ipv6:%5B::1%5D:50051 {grpc_status:13, grpc_message:""}"；

## Details

关于 IL Policy（如 ACT）的推理支持：
- [x] 解决一些很乱的规约、抽象 recorder 和 policy runner 共用代码：
    - [x] 重构 observation 定义，只保留 `observation.state`、`observation.images.*`、`task`；
    - [x] 修正 recorder 的 `action` 语义，改为记录可 replay 的绝对 joint target；对于 POSITION 控制直接记录 joint command，对于 velocity/twist 这类控制则先归一化，避免把不可 replay 的原始命令写入数据集；
    - [x] 将 LeRobot dataset 的“录制/回放 schema”和“推理输入 schema”解耦；
    - [x] `recorder_lerobot.py` 提取 backend snapshot -> LeRobot frame / observation 的公共转换逻辑；
    - [x] 为 MuJoCo backend 增加最小推理期 observation 构造路径，避免推理时重复走 recorder 的全量 feature 枚举；
    - [x] 明确 joint names 与 `JointModelGroup` 的映射规则，保证 action 能稳定下发到唯一 JMG；
- [x] IL policy runtime：
    - [x] 增加本地 policy runner，负责加载 checkpoint、preprocessor、postprocessor、policy；
    - [x] runner 支持 episode/reset 语义，在开始推理和 reset 后调用 `policy.reset()`；
    - [x] runner 支持固定 control loop，按给定频率执行 observation -> preprocess -> `select_action()` -> postprocess -> `set_joint_target()`；
- [x] 框架层布线：
    - [x] 新增独立的 policy inference service；
    - [x] 最小接口覆盖：加载 policy、开始推理、停止推理、查询当前状态；
    - [x] 服务层保证与 recorder / replay 互斥，避免同一 backend 实例上出现控制冲突；

## Tests
- [x] 为 observation / action 适配层补单元测试，覆盖 joint names、camera names、group 选择；
- [x] 为 runner 补单元测试，覆盖 queue reset、chunked action 消费、stop/start 状态切换；
- [x] 增加一条 MuJoCo Backend 的集成测试：加载本地 LeRobot policy stub 后至少能完成一次推理循环；
- [ ] 测试是否支持 ACT 这类标准 IL chunking policy；测试是否保持对其他 LeRobot IL policy 兼容；




# VERSION v0.0.4

## New Features
- [x] 以 function tools 和 MCP tools 的形式暴露 gRPC 接口；
- [x] 引入 Agent 模块控制；
- [x] 接入 `RobotDataService`，支持通过 gRPC 开始/结束 episode 录制；
- [x] 支持稳定版本的 LeRobot Dataset：实现 `LerobotDataRecorder`，按 `RecordOptions` 从 backend 采样并落盘到 LeRobotDataset v3；
- [x] 支持 MuJoCo 关节状态、末端位姿、控制目标、相机/IMU/LiDAR/Odom/力力矩传感器录制；
- [x] 为 Gazebo 补充最小 `GetRobotSpec`（实际功能例如真正解析 Gazebo 模型语义数据，有待后续补充），使 `jmg` 过滤能够工作；
- [x] 新增 `control_stubs/tools/servo_keyboard.py`，用终端键盘向 `ServoControlStream` 发送 ee twist / joint servo 调试命令；
- [x] 为 control tools / MCP tools 暴露录制接口；

## Bug Fixes
- [x] LeRobot image 模式下图片实际写入 parquet 后，清理 `images/` 下残留空目录，避免误判为 MuJoCo 相机采集失败；
- [x] 修复 MuJoCo 相机离屏渲染跨线程复用同一 `mujoco.Renderer`，导致 recorder 第一帧后出现黑帧/彩条；

## Tests
- [x] 补充 recorder / RobotDataService 单元测试；
- [x] 补充键盘 servo 客户端的绑定选择与命令构造单元测试；
- [x] 验证 MuJoCo server 启动路径；

# VERSION v0.0.3

## New Features
- [x] 实现 MuJoCoBackend；
- [x] 接入 MuJoCo 场景自动步进、headless/server 启动路径；
- [x] 实现 MuJoCo 的 jmg / ee / 传感器管理与基础重力补偿；
- [x] 补充 MuJoCo 单元测试与 pytest 收集范围配置；

## Bug Fixes
- [x] MuJoCo 空闲态未建立默认保持控制，导致启动后即使叠加抗重力也会快速塌落；

---

# VERSION v0.0.2

## New Features
- [x] 实现 GazeboBackend 的 navigate_to 能力；

## Bug Fixes

- [x] GazeboBackend 所有接口均无法正确读取数据（修复了名称匹配逻辑、添加了 Camera 格式化等）；
- [x] 在 GazeboBackend 中，所谓“动态传感器发现”只在第一次启动发现。确保能够定期更新传感器列表，注意并发问题（比如外界传感器信息更新后如果接口还在被读取的情况、能否确保 gRPC 接口始终读取到最新的数据等，计划加入测试进行检查）；
- [x] Ctrl-C 无法停止 gRPC 服务器（使用 `python3 -m robosim.server --port 50052` 启动）；

---

# VERSION v0.0.1

从旧版 OH RoboSim 精简重构。

## Phase 1: 基础设施
- [x] 项目结构和 proto 定义
- [x] 编译 proto 文件生成 Python stubs
- [x] 创建核心抽象层 (SimulatorBackend 基类)

## Phase 2: Gazebo 后端实现
- [x] 创建 GazeboBackend 类骨架
- [x] 实现动态传感器发现（按数据类型，非 topic 名称）
- [x] 实现通用能力检测（不依赖 Gazebo 专用模块）
- [x] 实现关节状态读取 (GetRobotState)
- [x] 实现关节目标设置 (SetJointTarget)
- [x] 实现传感器列表/读取 (ListSensors, GetSensors)

## Phase 3: 导航能力
- [x] 实现 GetRobotPoseInMap
- [x] 实现 NavigateTo (框架实现)

## Phase 4: 仿真控制
- [x] 实现 EmergencyStop

## Phase 5: gRPC 服务器
- [x] 创建 gRPC 服务器主入口
- [x] 集成所有服务

## Phase 6: 测试与质量
- [x] 编写单元测试 (14 tests passing)
- [x] 通过 ruff lint 检查
- [x] 通过 mypy 类型检查

## Phase 7: 文档
- [x] 更新 DESIGN.md

## 架构改进 (v2)
- [x] 移除对 `robot_sim_common` 模块的依赖
- [x] 动态发现传感器（按 ROS2 topic 数据类型）
- [x] 通用能力检测（通过运行时检查）
