# Habitat-Sim 后端启动指南

Habitat-Sim 后端和 Gazebo / MuJoCo 的启动路径差异较大：它依赖可选的
`habitat_sim` Python 包，显示窗口、headless camera renderer、Panda articulated object
分别走不同的 Habitat-Sim 运行模式。本指南集中说明 Habitat 后端的安装、启动和图像查看。

## 1. 准备 RoboSim 环境

先完成的通用环境创建和 proto 生成：

```bash
mamba env create -f environment.yml
mamba activate robosim
./scripts/gen_protos.sh --clean
./scripts/gen_protos.sh
```

若proto文件发生变化，需要重新运行`gen_proto.sh`

## 2. 安装 Habitat-Sim

Habitat-Sim 是可选依赖，不在 `environment.yml` 中自动安装。选择
`--backend habitat` 前，需要在当前 Python 环境中安装 `habitat_sim`。

可以使用仓库提供的源码安装脚本：

```bash
mamba activate robosim
./scripts/install_habitat.sh
```

脚本默认会：

- 从 `https://github.com/facebookresearch/habitat-sim.git` 拉取 `main` 分支；
- 在 `.tmp/habitat-sim/` 中构建；
- 安装 GUI viewer 和 Bullet 支持；
- 最后验证 `import habitat_sim`。

可用环境变量调整行为：

```bash
HABITAT_SIM_REF=v0.3.3 ./scripts/install_habitat.sh
HABITAT_SIM_BUILD_DIR=/tmp/habitat-sim-build ./scripts/install_habitat.sh
KEEP_HABITAT_BUILD_DIR=0 ./scripts/install_habitat.sh
ROBOSIM_ENV_NAME=robosim ./scripts/install_habitat.sh
```

## 3. 只启动 Habitat 场景渲染

不传入 `--robot` 时，Habitat 后端提供渲染能力，不控制机器人：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --scene /path/to/your/scene
```

`--scene` 需要传 Habitat-Sim 支持的 mesh / scene 文件，例如 `.glb`、`.gltf`、
`.obj` 或 `.ply`。

## 4. 本地显示窗口和软件渲染

在没有 NVIDIA GPU 但有显示器的环境中，可以使用普通显示版 Habitat-Sim 和 Mesa
软件渲染：

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=4.1 DISPLAY=:0 \
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --scene /path/to/your/scene \
  --no-headless
```

`--no-headless` 会打开 Habitat-Sim viewer 窗口。这个模式只适合直接查看单个 mesh 场景，
不会通过 gRPC 暴露 `habitat_rgb` 图像，也不能动态加载机械臂 URDF。

> [!TIP]
> 如果 `drivers_sim` 资产目录不在 robosim 仓库内，可以用`ROBOSIM_DRIVERS_SIM_ROOT` 指向外部资产根目录。`--scene` 支持相对该目录的路径：
>
> ```bash
> ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
> python3 -m robosim.server \
>   --port 50051 \
>   --backend habitat \
>   --scene habitat/assets/worlds/apartment.glb \
>   --no-headless
> ```

## 5. 加载机械臂

可以用 Habitat-Sim 的 articulated object API 加载机械臂：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot drivers_sim/gazebo-11/assets/robots/franka_panda \
  --headless
```

`--robot` 应传机器人资源目录或具体 `.urdf` 文件。Habitat 后端会在该目录下寻找 URDF
并加载；如果目录里只有 MuJoCo MJCF `.xml`，请改用 `--backend mujoco`。

Panda 模式会从机器人目录中的 Panda URDF 加载，并暴露 `panda_arm`、`panda_hand`、
`panda_arm_hand` 三个 joint model group。Habitat camera renderer 默认开启，因此可以直接
通过 sensing gRPC 接口读取 `habitat_rgb`。如果当前机器没有可用 GPU/EGL，可以用
`--no-habitat-enable-camera` 关闭 camera renderer，只保留 joint state/spec 和 POSITION
joint target。

在有 GPU/EGL 的机器上，可以同时加载环境场景和 Panda：

```bash
ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot gazebo-11/assets/robots/franka_panda \
  --scene habitat/assets/worlds/apartment.glb \
  --headless
```

Panda 可视化走 backend 的 camera renderer，不走 `--no-headless` 的 Habitat viewer
subprocess。后者只能直接打开单个 mesh 场景，不能动态加载 Panda URDF。

如果只想加载 Panda，不加载环境场景，可以去掉 `--scene`：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot drivers_sim/gazebo-11/assets/robots/franka_panda \
  --headless
```

## 6. 保存一帧 habitat_rgb

启动带 camera renderer 的 Habitat 后端后，另开一个终端，在同一个环境和仓库根目录下运行：

```bash
python3 save_habitat_rgb.py
```

生成的 `habitat_rgb.png` 会保存在当前目录。

也可以指定输出路径或服务地址：

```bash
python3 save_habitat_rgb.py --output /tmp/habitat_rgb.png
python3 save_habitat_rgb.py --host localhost --port 50051
```

## 7. 连续显示 habitat_rgb

如果安装了 OpenCV，可以直接使用仓库中的小工具连续显示图像流：

```bash
python3 move_habitat_camera.py
```

终端按 `a/d` 调整 yaw，`w/s` 调整 pitch，`r/f` 调整距离，按 `q` 或在窗口按 `Esc`
退出。

也可以保存一段视频或调整连接参数：

```bash
python3 move_habitat_camera.py --save-video habitat_rgb.mp4 --max-frames 120
python3 move_habitat_camera.py --host localhost --port 50051
```

更多参数：

```bash
python3 save_habitat_rgb.py --help
python3 move_habitat_camera.py --help
```

## 8. 常见问题

- `ModuleNotFoundError: No module named 'control_stubs.common_pb2'`：
  先运行 `./scripts/gen_protos.sh`。
- `Habitat-Sim backend requires the optional 'habitat_sim' package`：
  当前环境未安装 Habitat-Sim，先运行 `./scripts/install_habitat.sh` 或按 Habitat-Sim
  官方方式安装。
- `Habitat-Sim viewer cannot load MuJoCo MJCF XML scenes`：
  `--scene` 不能传 MuJoCo `scene.xml`。Habitat scene 要使用 `.glb`、`.gltf`、`.obj`
  或 `.ply`。
- `--no-headless` 看不到 Panda：
  这是预期行为。Panda 是通过 backend 的 Simulator API 动态加载的，只能走
  `--headless` 后通过 gRPC camera 查看。
- `--headless` 读取不到 `habitat_rgb`：
  headless camera renderer 需要可用 EGL/GPU 渲染上下文。无 GPU 的本地显示环境可以改用
  `--no-headless` 直接打开 viewer，但该模式不提供 gRPC camera 图像。如果只需要关节接口，
  可用 `--no-habitat-enable-camera` 关闭 camera renderer。
