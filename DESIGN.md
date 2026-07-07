# RoboSim 框架设计

## 目标
为多种模拟器后端（Gazebo/MuJoCo/PyBullet/Habitat-Sim）提供统一的控制和状态读取抽象，通过 gRPC 接口向上层暴露。

在 thesis-level benchmark generator 中，`vsim` 还承担 Concrete Scenario
Definition（CSD）的后端实现边界：上层 benchmark generator 负责从自然语言
benchmark distribution 采样并固化一个具体 CSD；`vsim` 负责将该 CSD
finalize/load 为 MuJoCo、Gazebo 或未来后端所需的 native scene/artifacts，
并继续提供渲染、传感器、控制、policy runtime 和 rollout 采集能力。CSD 在传入
`vsim` 时已经不是 distribution，而是一个具体 benchmark atom 的固定定义。

后端 native scene 生成不应放在 thesis-level benchmark generator 中。若在
`vsim` 内实现 MuJoCo/Gazebo 等后端的 CSD realization，必须先仔细阅读对应
官方文档；例如 MuJoCo 路径需要理解 MJCF 的 body、asset、geom、inertial、
joint、material、mesh、contact 参数以及 compiler defaults 等语义。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      gRPC Server                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │SimulationSvc│ │ SensingSvc  │ │RobotCoreSvc │ │Mobility│ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───┬────┘ │
└─────────┼───────────────┼───────────────┼────────────┼──────┘
          │               │               │            │
┌─────────▼───────────────▼───────────────▼────────────▼──────┐
│                   Backend Manager                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SimulatorBackend (Abstract Base)           │   │
│  │  + get_robot_state()                                 │   │
│  │  + get_robot_spec()                                  │   │
│  │  + set_joint_target()                                │   │
│  │  + get_end_effector_state()                          │   │
│  │  + list_sensors() / get_sensors() / stream_sensors() │   │
│  │  + navigate_to() / get_robot_pose_in_map()           │   │
│  │  + reset_world() / step_physics()                    │   │
│  │  + emergency_stop()                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              GazeboBackend (Concrete)                       │
│  - 使用 ROS2 topic 动态发现传感器                              │
│  - 订阅 JointState, Imu, LaserScan, Image 等                 │
│  - 发布 /cmd_vel 控制                                         │
│  - 使用 actionlib 调用 Nav2                                   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              MuJoCoBackend (Concrete)                       │
│  - 使用 MuJoCo Python Bindings 进行仿真                       │
│  - 从 XML 模型文件加载机器人配置                                │
│  - 支持关节位置/速度/扭矩控制                                   │
│  - 通过 forward kinematics 获取末端执行器位姿                   │
│  - 支持力/扭矩传感器、IMU 等                                    │
│  - 默认启用重力补偿                                            │
│  - 非 headless 模式支持被动 viewer                             │
└─────────────────────────────────────────────────────────────┘
```

## 模块结构

包含部分文件，仅作示意。

```
robosim/
├── core/                    # 核心抽象（后端无关）
│   ├── backend.py          # SimulatorBackend 基类
│   └── capabilities.py     # 能力枚举
├── backends/
│   ├── gazebo/            # Gazebo 后端实现
│   │   └── backend.py     # 主后端类
│   └── mujoco/            # MuJoCo 后端实现
│       └── backend.py      # 主后端类
├── grpc_server/           # gRPC 服务实现
│   ├── simulation.py
│   ├── sensing.py
│   ├── robot_core.py
│   └── mobility.py
└── server.py              # 主入口
```

## 设计细节

### 1. 后端无关的设计
- **不应依赖特定模拟器的配置模块**（如 `robot_sim_common.config`）
- 能力检测由后端自己实现，不依赖外部配置文件
- 每个后端独立发现其支持的传感器和运动接口

### 2. 传感器动态管理
- **不硬编码 topic 名称**
- 根据消息数据类型自动识别传感器（JointState, Imu, LaserScan, Image 等）
- 支持多传感器同时存在

### 3. 能力检测规约
- 通过检查实际可用的接口来判断能力
- 导航能力：检查 Nav2 action 或相关 topic 是否可用
- 伺服能力：检查 joint 命令接口是否可用
- 传感器能力：动态发现已订阅的传感器

### 4. Gazebo 后端 ROS Topics（动态发现）
| 数据类型 | Topic 模式 | 说明 |
|---------|-----------|------|
| JointState | `*/joint_states` | 关节状态 |
| Odometry | `*/odom` | 里程计 |
| Imu | `*/imu*` | IMU 传感器 |
| LaserScan | `*/scan*`, `*/laser*` | 激光雷达 |
| Image | `*/image_raw`, `*/compressed_image` | 相机 |

### 5. MuJoCo 后端能力检测
| 能力 | 检测方式 | 说明 |
|------|----------|------|
| JOINT_READ | 始终可用 | 从 qpos/qvel 读取关节状态 |
| JOINT_WRITE | 始终可用 | 支持位置/速度/扭矩控制 |
| NAVIGATION | 不可用 | MuJoCo 不直接支持导航 |
| SENSOR_* | 检查模型中的传感器/相机 | `<sensor>` + model camera 一起注册 |

### 6. 错误处理
- 不支持的操作用 `NotImplementedError` 标识
- gRPC 层将其映射为 `UNIMPLEMENTED` 状态码

## 关于 MuJoCoBackend

### 1. 运行时结构
- `mjModel` 只加载一次并保持只读；
- `mjData` 作为主仿真状态，由后端步进线程独占推进；
- 额外维护一个 `mjData` scratch 实例，用当前 `qpos` 和零速度重新 `mj_forward`，用于求控制环里的重力补偿项；
- 非 headless 模式使用 `viewer.launch_passive`，由后端在每次步进后主动 `sync(state_only=True)`。

### 2. 自动步进与控制
- MuJoCoBackend 默认启动后台线程自动步进，不依赖 `StepPhysics`；
- 步进循环采用 `mj_step1 -> 写入控制 -> mj_step2`，让控制计算落在 MuJoCo pipeline 允许的位置；
- 对于没有原生 actuator 的关节，使用 `qfrc_applied` 做关节级控制；
- 默认空闲态会先落到 SRDF 默认姿态，再做 position hold 并持续叠加抗重力项；
- 位置/速度模式在关节空间内转成简洁的 PD 力矩控制，并叠加重力补偿；
- 扭矩模式直接写目标力矩，并叠加重力补偿。

### 3. jmg / ee 语义
- 优先读取 `.srdf`；
- 由于当前环境中的 `srdfdom` 对 `passive_joint` 支持不完整，实际实现使用 XML 解析保留完整语义；
- group 支持 `chain`、`joint`、`passive_joint`、`link`、子 group 展开；
- 若没有 SRDF，则把同一 MuJoCo kinematic tree 的全部非 free/ball 关节收为一个 jmg；
- end effector 优先绑定同名 body/site，否则回落到 group tip body。
- 若 SRDF 中存在 `home/ready/default` 这类命名状态，启动和 reset 时优先应用。

### 4. 传感器管理
- MuJoCo 传感器不是 topic，因此后端在初始化时建立静态 registry；
- registry 同时包含：
  - `joint_states` 伪传感器；
  - MJCF `<sensor>` 中可直接映射的传感器；
  - model cameras；
- camera 通过 `mujoco.Renderer` 做 offscreen rendering；
- MuJoCo 的 offscreen renderer 绑定创建它的线程；后端按“线程 + 分辨率”缓存 renderer，避免录制线程和 gRPC 请求线程跨线程复用 EGL/OpenGL 上下文导致黑帧/花屏；
- 当前 gRPC 扩充了力/力矩传感器类型与数据结构，以避免 MuJoCo 现有传感器语义丢失。

## 关于 LeRobot 数据录制

### 1. 录制入口
- `RobotDataService`，仅暴露 `EpisodeStart` / `EpisodeEnd`；
- gRPC server 启动时构造一个 `LerobotDataRecorder`，数据根目录固定为仓库下 `data/lerobot/<repo_name>`；
- 同一 backend 实例同一时刻只允许一个录制 session。

### 2. 采样模型
- `EpisodeStart` 读取一次 backend 的 `robot_state`、`robot_spec`、`end_effector_state`、`sensor snapshot`，先推导数据集 schema，再立即写入第一帧；
- 后续由后台线程按 `fps` 继续采样；
- `EpisodeEnd` 停止采样线程、保存 episode，并立即 `finalize()`，保证每次结束后磁盘上的数据集都是可加载状态。

### 3. 字段映射
- 关节状态统一映射为：
  - `observation.state`
  - `observation.velocity`
  - `observation.effort`
- `action` 固定定义为 joint coordinate system 下的绝对关节位置目标，记录 backend 最近一次 joint command 的 position target，保证 replay 与 policy runtime 共享同一动作语义；
- 末端位姿映射为 `observation.end_effectors.<group>.{position,orientation}`；
- 视觉数据映射为 `observation.images.<sensor>`；
- IMU / LiDAR / Odom / Force / Torque 各自映射到对应 `observation.*` 数值向量。

### 4. 过滤规则
- `jmg_excluded` 先于 `jmg_included` 生效；
- 未显式指定 `jmg_*` 时，默认录制当前 `get_robot_state()` 返回的全部关节；
- 未显式指定 `sensor_*` 时，默认录制全部非 joint 传感器；
- 现阶段不重复把 joint sensor 写入 dataset，因为它和 `observation.state` 语义重叠。

### 5. LeRobot 落盘策略
- 直接复用 `lerobot>=0.5` 提供的 `LeRobotDataset.create()/resume()/add_frame()/save_episode()/finalize()`；
- 当前相机帧使用 `image` 特征直接写盘，不走 mp4 编码，先保证 v3 数据结构稳定和测试确定性；
- `lerobot` 会在写 parquet 时把图片字节嵌入 parquet，再删除 `images/<camera>/episode-*` 下的临时 PNG；recorder 在 episode 结束后继续清理残留空目录，避免留下误导性的空相机目录；
- 已存在 repo 继续录制时使用 `resume()`，并校验 `fps` 与 feature schema 必须完全一致。

### 6. 数据集重放
- `RobotDataService` 额外暴露 `EpisodeReplay(repo_name, episode_id)`，语义是阻塞式单 episode 回放；
- 回放时从本地 `data/lerobot/<repo_name>` 打开 LeRobotDataset，并通过 `episodes=[episode_id]` 只读取目标 episode；
- 回放只消费 dataset 的 `action` 向量，固定以 `JointCommand.POSITION` 下发到 backend，因此 replay 语义与 LeRobot 常见 joint-space absolute action 一致；
- 为避免 `all` / 子 group 同时包含相同 joint 集合导致 MuJoCo `set_joint_target()` 组选择歧义，回放前先读取当前 `RobotSpecification`，优先选择 joint 列表精确匹配的最小 joint model group；
- Gazebo 当前不实现回放写入路径，相关方法允许直接 `NotImplementedError`。

## 关于 LeRobot IL 推理支持

### 1. 当前范围
- 当前阶段只支持 MuJoCo 后端上的 LeRobot IL policy 推理；
- 不实现训练支持；
- 首个目标是兼容 ACT 这类标准 joint-space chunking policy，但运行时适配层应尽量保持对其他 LeRobot IL policy 通用。

### 2. 录制 / 回放 / 推理
- replay 继续消费 dataset 中的 `action`，并按 joint-space absolute position target 下发；
- 推理期的 observation 不直接复用 recorder 的“全量 feature schema”作为 policy 输入；
- 推理期只暴露最小必需 observation：`observation.state`、`observation.images.*`、`task`；
- recorder 允许继续记录更丰富的 feature，但这不应反向决定 policy runtime 的输入结构。

### 3. action 语义
- 框架内 LeRobot policy runtime 的标准 action 语义固定为 joint-space absolute position target；
- recorder 的 `action` 应记录“可 replay 的绝对关节目标”；
- backend 若内部直接运行 POSITION 控制，则 `action` 可直接取最近一次 joint command；
- backend 若当前控制源是 VELOCITY / TORQUE / twist 等非绝对位置命令，则必须先归一化成绝对关节位置目标再暴露给 recorder；至少不能把这类原始命令直接写进 `action`，否则 replay 按 POSITION 下发时会失真。

### 4. 运行时结构
- server 内独立的 policy runner，职责仅限于：
  - 加载 checkpoint / preprocessor / postprocessor / policy；
  - 维护推理线程与 stop/reset 状态；
  - 在固定控制频率下执行 observation -> preprocess -> `select_action()` -> postprocess -> `set_joint_target()`；
  - 在开始新一轮推理或 world reset 后调用 `policy.reset()`，清理 LeRobot policy 内部的 action queue / history。
- policy runner 与 recorder / replay 必须互斥，同一 backend 实例任一时刻只能存在一种主动控制源。

### 5. Observation 适配
- backend snapshot -> LeRobot observation 的转换逻辑应从 recorder 中抽离成共享组件；
- 该适配层负责：
  - 选择推理需要的 joints / cameras；
  - 生成与 policy checkpoint 对齐的 observation keys；
  - 维持 joint names 与 `JointModelGroup` 的稳定映射；
  - 在 postprocess 后把 action 向量重新映射为 backend 可执行的 joint target。

## 关于 ServoControlStream 调试客户端

- 调试客户端位于 `control_stubs/tools/servo_keyboard.py`，职责仅限于把终端键盘事件转成 `ServoCommand` 流，不引入新的控制抽象；
- 客户端启动时先读取 `RobotSpecification`，默认自动选择一个带 end effector 的 `jmg` 作为 twist 控制目标；若存在较小的非 ee `jmg`，则同时把它作为 joint 调试目标；
- 终端 raw mode 只能稳定拿到 key-down，不能可靠拿到 key-up，因此客户端采用“按键生效一小段保持时间，超时后自动补发零速度”的语义，避免后端持续保持旧 velocity target；
- `ServoCommand` 是 `oneof`，因此 twist 和 joint 调试命令始终分开发送，客户端每个周期最多发一条 twist 命令和一条 joint 命令。
