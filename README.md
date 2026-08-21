
## RoboSim: A Meta-Simulator Framework for Embodied Intelligence

### Supported Features

- [x] 支持 Gazebo / MuJoCo / PyBullet 模拟器后端；

- [x] 支持动态的传感器发现、机器人关节和定义发现；

- [x] 语言无关的控制接口 (gRPC)，无需用户了解模拟器细节。包括 core（机器人及环境状态查询和操作）、sensing（传感器相关操作）、simulation（仿真环境相关操作）、mobility（AI/导航）、data（仿真数据收集）、policy（IL/RL policy 推理）共 6 个方面；

- [x] 简单的示例，包括一个 [multi-modal agent](./agent/README.md)，各种[实用工具](./control_stubs/tools/) 例如配套的 gRPC client (python)，键盘伺服操纵工具，function tools，MCP tools 等；

- [ ] (WIP) 多类精彩的模拟仿真环境 `drivers_sim`；

- [ ] (WIP) 对真实机器人本体（如曦胧本体）的适配 `drivers_real`；

- [ ] (WIP) 支持基于 IL/RL 训练的 Policy 的推理过程；


### CSD 编译与运行

`vsim` 接收固定的 OpenUSD CSD package，并为 MuJoCo、Gazebo 或 PyBullet 生成可重复的本地 realization。CSD 是场景语义源；MJCF、SDF、URDF 和 Python scene 文件都是派生产物，不能反向作为场景定义。

命令行入口使用 `scene-export/v9-vsim-articulated-resources` package。输入为 package 根目录的 `scene.usda`、同级 `manifest.json`、checksum 与资产 dependency closure。仓库中可直接使用 `csd/benchmark_gen/scene.usda`。

#### 编译

```bash
python -m robosim.compile \
  --csd csd/benchmark_gen/scene.usda \
  --backend gazebo \
  --output-root csd/benchmark_gen/engine_manifests
```

`--backend` 可选 `mujoco`、`gazebo` 或 `pybullet`。成功时标准输出为 manifest JSON；
无法保持语义或缺少运行时依赖时，输出 typed blocker 并以退出码 `2` 结束。

输出固定写入 `<output-root>/<backend>/<csd_id>/`，其中包含 `manifest.json`、后端
entry file、`assets/` 和 `diagnostics/`。实际使用的 mesh、texture 和 robot dependency
closure 会复制到 realization package；运行时不得依赖 `drivers_sim` 或下载 cache。

MuJoCo、Gazebo 与 PyBullet 分别生成 `scene.xml`、`world.sdf` 与 `scene.py`；Python
API `compile_csd()` 返回 manifest 或 typed blocker。


### Quick Start

拉下本仓库并准备环境。以 miniforge 管理虚拟环境为例：

```bash
git clone --recursive https://github.com/SSRVodka/robosim.git
pushd robosim
mamba env create -f environment.yml
mamba activate robosim
popd
```

> [!NOTE]
> 
> 如果下拉仓库时没有添加 `--recursive` 选项，可以执行下面的命令来补充拉取子模块资产：
> 
> ```bash
> git submodule update --init --recursive
> ```

再编译 proto 接口（一旦存在 *.proto 文件的更新就需要重新执行）：

```bash
# 需要在 robosim 环境下，即 mamba activate robosim，下面不再赘述
./scripts/gen_protos.sh --clean
./scripts/gen_protos.sh
```

最后启动 robosim（`[]` 表示可选项，`<>` 表示必填项）。更多参数用法请使用 `--help`：

```bash
python3 -m robosim.server [--help] [--port <gRPC-listen-port>] [--backend <gazebo|mujoco|pybullet>] [--headless | --no-headless] [ --csd-manifest </path/to/manifest> ]
```

> [!WARNING]
>
> Gazebo legacy 模式（不传 `--csd-manifest`）仍连接用户自行启动的 ROS/Gazebo graph。
> manifest 模式由 server 完整管理 Gazebo Classic 与 controllers，无需另开 runtime。
>
> ```bash
> mamba activate robosim
> pushd drivers_sim/gazebo-11/  # Jazzy / Harmonic 使用 drivers_sim/gazebo/
> colcon build
> source ./install/setup.bash
> popd
> ros2 launch demos gzsim.nav2.launch.py gui:=false
> ```
>
> 如果使用编译出的 Gazebo realization，启动流程为：
>
> ```bash
> mamba activate robosim
> python -m robosim.server --port <gRPC-listen-port> \
>   --backend gazebo --csd-manifest </path/to/manifest> \
>   [--headless | --no-headless]
> ```


现在，你的环境已经准备好了！

PyBullet 和 MuJoCo 后端不需要额外启动 ROS2 节点。headless 模式使用 PyBullet DIRECT client；`--no-headless` 使用 GUI client。MuJoCo、PyBullet 和 Gazebo 的原生场景查看均统一由 server 的 `--no-headless` 提供。

> (WIP) OpenHarmony 部署环境的文档正在准备中。


### 实用工具演示

#### A. 本框架如何接入 Agent

`control_stubs/tools/` 给出了 gRPC 的 client 定义、function tools 和 MCP tools 定义，你可以用它们接入任何主流的 Agent 框架中作为 Agent Tools 使用。

> [!TIP]
> 
> 当然本项目也提供了一个最小化的示例 Agent 实现，实现细节参见 [`agent/README.md`](./agent/README.md)。您可以按照 `agent/config/default.yaml` 中写一份配置，然后使用 `agent_orchestrator.py` 来尝试。
> 
> 确保您的 shell 在仓库根目录下。在启动 robosim gRPC server 后执行下面的指令进入 Agent REPL（使用 `--help` 查看帮助）：
> 
> ```bash
> python3 agent_orchestrator.py --config <你的配置文件> --grpc-host 127.0.0.1 --grpc-port <你之前robosim启动设置的端口> chat
> ```

#### B. 简单的测试伺服操作 demo

确保您的 shell 在仓库根目录下。

以 MuJoCo 后端为例，先启动 robosim（需要确保您的宿主机环境支持 OpenGL）：

```bash
python3 -m robosim.server --port 50051 --backend mujoco --no-headless
```

此时会弹出模拟环境 GUI。然后使用伺服工具查看现在有哪些关节和关节组能被伺服控制：

```bash
python3 -m control_stubs.tools.servo_keyboard --list
```

例如如果输出是这样的：

```
robot: panda
  panda_arm: joints=7 ee=hand
  panda_hand: joints=2 ee=-
  panda_arm_hand: joints=9 ee=-
```

表示当前可以操纵的关节模型组有 3 个，其中 `panda_arm` 这个组存在一个末端执行器 `hand`。

您可以在笛卡尔坐标系下通过键盘驱动末端执行器：

```bash
python3 -m control_stubs.tools.servo_keyboard --twist-target panda_arm:hand
```

现在您的终端应该打印消息提示如何操纵这个关节模型组了。根据提示操纵即可。

更多能力，例如直接操纵指定关节位置/速度/力矩、调整指令发送的频率等等，请参见工具的 `--help` 信息：

```bash
python3 -m control_stubs.tools.servo_keyboard --help
```

统一 teleop 入口允许 Cartesian 和 direct-joint 两类 JMG 同时保持活动，并可分别配置多个候选目标。例如下面的键盘会同时控制 Panda 手臂和夹爪：

```bash
python3 -m control_stubs.tools.teleop \
  --input keyboard \
  --twist-target panda_arm:hand \
  --joint-target panda_hand
```

`--twist-target GROUP[:EE]` 和 `--joint-target GROUP` 均可重复使用。运行期间按 `n` 切换 Cartesian target，按 `m` 切换 direct-joint target；切换时客户端会先清零旧 target。省略这些参数时，候选项由 `GetRobotSpec` 自动发现，因此实现不绑定  Franka Panda 或特定的 arm/gripper 配置。这两个参数是唯一的 servo target 选择接口。

> [!TIP]
>
> 对于 `unitree_g1`，`drivers_sim/mujoco/assets/robots/unitree_g1/scene.xml`为上半身双臂模型。如需测试29dof的全身模型，请使用`python3 -m robosim.server --port 50051 --backend mujoco --no-headless --scene assets/robots/unitree_g1/g1_29dof.xml`来启动 robosim。

#### C. 简单的测试 LeRobot 数据采集 & 重放 demo（命令行）

确保您的 shell 在仓库根目录下。

以 MuJoCo 后端为例，先启动 robosim（需要确保您的宿主机环境支持 OpenGL）：

```bash
python3 -m robosim.server --port 50051 --backend mujoco --no-headless
```

推荐使用统一 teleop 入口进行快速采集。指定 `--repo-name` 后客户端会立即开始首个 episode；以下键盘示例同时记录手臂与夹爪：

```bash
python3 -m control_stubs.tools.teleop \
  --input keyboard \
  --twist-target panda_arm:hand \
  --joint-target panda_hand \
  --repo-name demo1 \
  --task-text "pick and place" \
  --fps 30
```

键盘保留上一节的 motion keys，并使用 `[` / `]` 控制 direct-joint target；`e` 保存当前 episode 并开始下一个，`c` 丢弃当前 episode 并立即重试，`space` 清零 motion，`q` 丢弃未完成 episode 并退出。

right Joy-Con 通过参数显式启用。`environment.yml` 已包含 `evdev`；若只安装 Python package，可使用 `pip install -e '.[joycon]'`。当前本机设备的采集命令为：

```bash
python3 -m control_stubs.tools.teleop \
  --input joycon \
  --input-device /dev/input/event15 \
  --input-profile joycon-right \
  --twist-target panda_arm:hand \
  --joint-target panda_hand \
  --repo-name demo1 \
  --task-text "pick and place" \
  --fps 30 \
  --reset-between-episodes
```

right Joy-Con profile 的 stick 控制 Cartesian X/Y，R/ZR 控制 Z；按住 SL 后 stick 控制 roll/pitch，R/ZR 控制 yaw；X/Y 控制 direct-joint target，可用于夹爪。

Home 和 stick press 分别切换两类 target，A 保存，B 丢弃并重试，Plus 丢弃并退出。

`--reset-between-episodes` 是可选参数。开启时顺序固定为 save/cancel → `ResetWorld` → next start；不开启时直接开始下一个 episode。MuJoCo 与 PyBullet 的 reset 可恢复初始 scene（目前忽略 `seed` 和 `randomization_params`）；Gazebo 尚未实现 reset，因此会返回 `UNIMPLEMENTED` 并停止采集。

数据存放在 `data/lerobot/demo1`。自动化测试使用 synthetic evdev events 和 fake clients，不访问 `/dev/input/event*`；上述映射已在 `/dev/input/event15` 手工确认。设备重新连接后 event 编号可能变化，此时只需更新 `--input-device`。

如需把录制与控制拆开，仍可直接操作 episode RPC：

```bash
python3 -m control_stubs.tools.data_recorder start --repo-name demo1 --task-text "demo-move"
python3 -m control_stubs.tools.data_recorder end     # 保存
python3 -m control_stubs.tools.data_recorder cancel  # 丢弃
```

> [!TIP]
>
> 默认存放在项目根目录下的 `data/lerobot` 中，您可以通过更改 `robosim/server.py` 中的 `DATA_REPO_ROOT` 变量来决定以何目录为数据根目录；


如需重放数据，需确保数据位于 `data/lerobot`，然后指定 repo name 和 episode ID：

```bash
python3 -m control_stubs.tools.data_recorder replay --repo-name demo1 --episode-id 0
```

---

## 开发规约与环境说明

- robosim 环境提供了 `ruff` 和 `mypy`。在 PR/提交前需要通过 `ruff` 和 `mypy` 的 lint 检查。之后我会设置 pre-commit hooks；
- 本项目开发环境统一使用 miniforge 管理的虚拟环境；
- robosim 环境已经提供了固定 gRPC 的版本（`grpcio==1.78.1`,`protobuf==6.33.5`），不得随意更改这个版本，这提供了对 OpenHarmony ArkUI 的兼容性；


## 仿真环境资产规约

详细请参见 [`drivers_sim`](./drivers_sim/README.md)；
