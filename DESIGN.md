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
│  - 使用 MuJoCo Python bindings 直接访问物理引擎               │
│  - 通过传感器系统获取 IMU、力/力矩、速度等数据               │
│  - 直接读写关节位置/速度进行伺服控制                         │
│  - 通过正向运动学获取末端执行器位姿                           │
│  - 无导航支持（纯伺服控制）                                  │
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

### 5. MuJoCo 后端设计
| 特性 | 实现方式 |
|-----|---------|
| 关节控制 | 直接修改 `data.qpos`/`data.qvel` |
| 传感器 | 通过 `model.sensor_*` 和 `mj_sensorPos` 获取 |
| 末端执行器 | 通过 `xpos`/`xquat` 进行正向运动学 |
| 能力 | 关节读写、传感器读取、紧急停止 |

### 6. 错误处理
- 不支持的操作用 `NotImplementedError` 标识
- gRPC 层将其映射为 `UNIMPLEMENTED` 状态码

## 模块结构
```
robosim/
├── core/                    # 核心抽象（后端无关）
│   ├── backend.py          # SimulatorBackend 基类
│   └── capabilities.py     # 能力枚举
├── backends/               # 后端实现
│   ├── gazebo/            # Gazebo 后端实现
│   │   └── backend.py
│   └── mujoco/           # MuJoCo 后端实现
│       └── backend.py
├── grpc_server/            # gRPC 服务实现
│   ├── simulation.py
│   ├── sensing.py
│   ├── robot_core.py
│   └── mobility.py
└── server.py              # 主入口
```
