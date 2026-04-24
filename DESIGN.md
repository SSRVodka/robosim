# RoboSim 框架设计

## 目标
为多种模拟器后端（Gazebo/MuJoCo/PyBullet/Habitat-Sim）提供统一的控制和状态读取抽象，通过 gRPC 接口向上层暴露。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      gRPC Server                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │SimulationSvc│ │ SensingSvc  │ │RobotCoreSvc │ │Mobility│ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───┬────┘ │
└─────────┼──────────────┼──────────────┼─────────────┼──────┘
          │              │              │             │
┌─────────▼──────────────▼──────────────▼─────────────▼──────┐
│                   Backend Manager                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SimulatorBackend (Abstract Base)             │   │
│  │  + get_robot_state()                                  │   │
│  │  + get_robot_spec()                                  │   │
│  │  + set_joint_target()                                │   │
│  │  + get_end_effector_state()                          │   │
│  │  + list_sensors() / get_sensors() / stream_sensors() │   │
│  │  + navigate_to() / get_robot_pose_in_map()          │   │
│  │  + reset_world() / step_physics()                    │   │
│  │  + emergency_stop()                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              GazeboBackend (Concrete)                          │
│  - 使用 ROS2 topic 动态发现传感器                             │
│  - 订阅 JointState, Imu, LaserScan, Image 等                 │
│  - 发布 /cmd_vel 控制                                       │
│  - 使用 actionlib 调用 Nav2                                  │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              MuJoCoBackend (Concrete)                          │
│  - 使用 MuJoCo Python Bindings 进行仿真                       │
│  - 从 XML 模型文件加载机器人配置                               │
│  - 支持关节位置/速度/扭矩控制                                  │
│  - 通过 forward kinematics 获取末端执行器位姿                    │
│  - 支持力/扭矩传感器、IMU 等                                  │
│  - 默认启用重力补偿                                           │
│  - 非 headless 模式支持被动 viewer                           │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计决策

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

## MuJoCo 设计补充

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
- 当前 gRPC 扩充了力/力矩传感器类型与数据结构，以避免 MuJoCo 现有传感器语义丢失。

## 模块结构
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
