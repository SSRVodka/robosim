# TODO - 短期里程碑


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
- [ ] 支持常见模型的 VLA 推理（对接 LeRobot 接口）；

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
