
## RoboSim: A Meta-Simulator Framework for Embodied Intelligence

### Supported Features

- [x] 支持 Gazebo / MuJoCo / PyBullet 模拟器后端；

- [x] 支持动态的传感器发现、机器人关节和定义发现；

- [x] 语言无关的控制接口 (gRPC)，无需用户了解模拟器细节。包括 core（机器人及环境状态查询和操作）、sensing（传感器相关操作）、simulation（仿真环境相关操作）、mobility（AI/导航）、data（仿真数据收集）、policy（IL/RL policy 推理）共 6 个方面；

- [x] 简单的示例，包括一个 [multi-modal agent](./agent/README.md)，各种[实用工具](./control_stubs/tools/) 例如配套的 gRPC client (python)，键盘伺服操纵工具，function tools，MCP tools 等；

- [ ] (WIP) 多类精彩的模拟仿真环境 `drivers_sim`；

- [ ] (WIP) 对真实机器人本体（如曦胧本体）的适配 `drivers_real`；

- [ ] (WIP) 支持基于 IL/RL 训练的 Policy 的推理过程；


### CSD -> Backend Scene Compiler

`vsim` exposes the CSD compiler boundary through `robosim.core.compile_csd`.
Pass `backend="mujoco"`, `backend="gazebo"`, or `backend="pybullet"`. The compiler consumes a fixed
Concrete Scenario Definition, an asset registry with passed backend variants,
an output root, and an asset root. In benchmark packages, pass
`output_root=Path("<package>/engine_manifests")`. The MuJoCo target writes
`engine_manifests/mujoco/<csd_id>/scene.xml`; the Gazebo target writes
`engine_manifests/gazebo/<csd_id>/world.sdf`; the PyBullet target writes
`engine_manifests/pybullet/<csd_id>/scene.py` plus `scene_meta.json` and
package-local URDF/assets. Backend targets copy referenced
assets under the backend artifact's local `assets/` directory, then return a
`CsdCompilationResult` containing either a `CsdRealizationManifest` or typed
`CsdRealizationBlocker` records.

The current compiler scope is intentionally narrow: rigid mesh objects with CSD
poses, backend mesh variants addressed by relative paths under `asset_root`,
optional MuJoCo `freejoint` for non-static objects, Gazebo SDF
model/link/visual/collision elements, and scalar mass/friction hints from object
`initial_state`. Runtime loading, render previews, and physics validation remain
separate follow-up stages.

Generated backend artifact directories are self-contained for the mesh variants
they use. MuJoCo `scene.xml` points `compiler meshdir` at the copied local
`assets/` directory. Gazebo `world.sdf` uses SDFormat 1.12 mesh URIs such as
`assets/objects/mug.obj`; the compiler does not require a ROS2 package, launch
directory, or package share layout. PyBullet realization treats the full
package as the backend scene: URDF files represent bodies, `scene.py`
deterministically assembles the physics world through PyBullet APIs, and
`scene_meta.json` records sensors, cameras, and CSD entity mappings.

The next MuJoCo compiler target is a complete realization package:

```text
engine_manifests/
  mujoco/
    <csd_id>/
      manifest.json
      scene.xml
      assets/
      diagnostics/
```

`scene.xml` must be loadable from that directory without depending on the
source `drivers_sim` tree or download caches. Existing `drivers_sim` robot/world
assets may be used as temporary template sources, but the compiler must copy
their required dependency closure into the realization directory before
referencing them from generated MJCF.

```python
from pathlib import Path

from robosim.core import compile_csd

result = compile_csd(
    backend="mujoco",
    csd=csd_json,
    asset_registry=asset_registry_json,
    output_root=Path("engine_manifests"),
    asset_root=Path("assets"),
)

if result.manifest is None:
    print([blocker.to_json_dict() for blocker in result.blockers])
else:
    print(result.manifest.to_json_dict())
```


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
python3 -m robosim.server [--help] [--host <gRPC-listen-host>] [--port <gRPC-listen-port>] [--backend <gazebo|mujoco|pybullet>] [--headless | --no-headless]
```

> [!WARNING]
>
> 如果选择的后端是 gazebo，那么需要额外启动 ROS2 节点（后续会集成进 `server.py`）。需要先在新的窗口中使用 robosim 虚拟环境：
>
> ```bash
> mamba activate robosim
> pushd drivers_sim/gazebo-11/
> # 解压 gazbo 预设模型
> tar -zxpvf assets-model.tar.gz
> # 构建 Gazebo 项目
> colcon build
> source ./install/setup.bash
> popd
> mamba activate robosim
> ros2 launch demos gzsim.nav2.launch.py
> ```
>
> 然后再启动 robosim。

现在，你的环境已经准备好了！

PyBullet 后端不需要额外启动 ROS2 节点。headless 模式使用 PyBullet DIRECT
client；`--no-headless` 使用 GUI client。

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
python3 -m control_stubs.tools.servo_keyboard --jmg panda_arm --ee hand
```

现在您的终端应该打印消息提示如何操纵这个关节模型组了。根据提示操纵即可。

更多能力，例如直接操纵指定关节位置/速度/力矩、调整指令发送的频率等等，请参见工具的 `--help` 信息：

```bash
python3 -m control_stubs.tools.servo_keyboard --help
```

> [!TIP]
>
> 对于unitree_g1，`drivers_sim/mujoco/assets/robots/unitree_g1/scene.xml`为上半身双臂模型。如需测试29dof的全身模型，请使用`python3 -m robosim.server --port 50051 --backend mujoco --no-headless --scene assets/robots/unitree_g1/g1_29dof.xml`来启动 robosim。

#### C. 简单的测试 LeRobot 数据采集 & 重放 demo（命令行）

确保您的 shell 在仓库根目录下。

以 MuJoCo 后端为例，先启动 robosim（需要确保您的宿主机环境支持 OpenGL）：

```bash
python3 -m robosim.server --port 50051 --backend mujoco --no-headless
```

执行下面的指令开始录制数据：

```bash
python3 -m control_stubs.tools.data_recorder start --repo-name demo1 --task-text "demo-move"
# 更多配置选项请使用
# python3 -m control_stubs.tools.data_recorder start --help
```

数据采集期间您可以使用各种方法操作仿真环境的机器人（例如使用上一节提到的 “测试伺服操作 demo”）。

执行下面的指令结束录制并将数据落盘：

```bash
python3 -m control_stubs.tools.data_recorder end
```

现在您的数据存放在 `data/lerobot/demo1` 下。如需重放该数据，请继续往下看。

> [!TIP]
>
> 默认存放在项目根目录下的 `data/lerobot` 中，您可以通过更改 `robosim/server.py` 中的 `DATA_REPO_ROOT` 变量来决定以何目录为数据根目录；


如需重放数据，需确保您采集的数据放在 `data/lerobot` 下，这样我们给定数据集的 `repoName` 以及 eposide ID 即可重放该数据集：

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
