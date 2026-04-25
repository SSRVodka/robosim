# MuJoCo 场景资产说明

本目录保留的是 MuJoCo 场景运行所需的最小资产集，只包含：

- `worlds/` 下的场景入口与场景本地 mesh / texture 资源
- `robots/` 下被场景直接引用的机器人模型资源

本目录不包含 `launch`、`demos`、`config`、ROS2 bridge 或地图生成脚本等外围运行层文件。

## 使用方式

以下命令默认在仓库根目录执行：

```bash
cd ~/Desktop/ros_oh/robosim
```

如果当前 Python 环境已经安装 MuJoCo Python bindings，可以直接用官方 viewer 打开场景：

## 场景清单

### bedroom

- 入口文件：`drivers_sim/mujoco/assets/worlds/bedroom/scene.xml`
- 依赖关系：单文件场景，不依赖额外机器人或外部 mesh/texture 目录

```bash
python3 -m mujoco.viewer --mjcf drivers_sim/mujoco/assets/worlds/bedroom/scene.xml
```

### cafe

- 入口文件：`drivers_sim/mujoco/assets/worlds/cafe/scene.xml`
- 依赖关系：依赖 `drivers_sim/mujoco/assets/worlds/cafe/assets/` 下的本地机械臂 mesh 资源
- 说明：该场景把 Panda 所需 mesh 打包在 world 本地目录内，不依赖 `robots/franka_panda`

```bash
python3 -m mujoco.viewer --mjcf drivers_sim/mujoco/assets/worlds/cafe/scene.xml
```

### oldroom

- 入口文件：`drivers_sim/mujoco/assets/worlds/oldroom/scene.xml`
- 依赖关系：依赖 `drivers_sim/mujoco/assets/robots/vx300s_cohesive/` 与 `drivers_sim/mujoco/assets/worlds/oldroom/git kelong/` 下的家具与纹理资源

```bash
python3 -m mujoco.viewer --mjcf drivers_sim/mujoco/assets/worlds/oldroom/scene.xml
```

### two_bedroom_apartment

- 入口文件：`drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/scene.xml`
- 依赖关系：依赖 `drivers_sim/mujoco/assets/robots/robot_vacuum/` 与 world 目录下的 OBJ / PNG 资产

```bash
python3 -m mujoco.viewer --mjcf drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/scene.xml
```

## 迁移说明

这批迁移到当前仓库的 MuJoCo 最小资产集包括：

- `drivers_sim/mujoco/assets/worlds/bedroom`
- `drivers_sim/mujoco/assets/worlds/cafe`
- `drivers_sim/mujoco/assets/worlds/oldroom`
- `drivers_sim/mujoco/assets/worlds/two_bedroom_apartment`
- `drivers_sim/mujoco/assets/robots/robot_vacuum`
- `drivers_sim/mujoco/assets/robots/vx300s_cohesive`

静态引用已经检查过，以上四个 `scene.xml` 中的 `include`、`mesh`、`texture` 路径在当前仓库内均可解析。
