# TODO - RoboSim 项目精简版重构短期里程碑

# VERSION v0.0.2

## New Features
- [x] 实现 GazeboBackend 的 navigate_to 能力。
- [x] 实现 MuJoCoBackend；
- [x] Agent 实现接入；
- [ ] Agent 集成测试；
- [ ] 接入 LeRobot 数采；

## Bug Fixes

- [ ] MuJoCo 关节无法正确使用 ServoControl，数据时间戳错误、 joint values 恒为 0；
- [ ] LeRobot 数采接口和推理相关接口变更，需要适配；
- [x] GazeboBackend 所有接口均无法正确读取数据（修复了名称匹配逻辑、添加了 Camera 格式化等）；
- [x] 在 GazeboBackend 中，所谓“动态传感器发现”只在第一次启动发现。实际上需要定期更新传感器列表，**注意并发问题（比如外界传感器信息更新后如果接口还在被读取的情况、能否确保 gRPC 接口始终读取到最新的数据等，需要加入测试进行检查）**；
- [x] Ctrl-C 无法停止 gRPC 服务器（使用 `python3 -m robosim.server --port 50052` 启动）。以实际检查这个现象不再出现为验收标准。

---

# VERSION v0.0.1

## Phase 1: 基础设施
- [x] 理解项目结构和 proto 定义
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

## Bug Fixes
- [x] 移除对 `robot_sim_common` 模块的依赖
- [x] 动态发现传感器（按 ROS2 topic 数据类型）
- [x] 通用能力检测（通过运行时检查）
