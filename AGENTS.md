# AGENTS.md

## 项目概述
本项目是使用 Python3 编写的元模拟器框架，作用是向上提供模拟器控制和状态读取的统一抽象（传感器读取、伺服驱动、导航能力），通过实现无关的 gRPC 接口暴露（接口参见 `control_stubs/*.proto`）；向下封装和管理不同的模拟器后端实例。

## 代码规范和红线
- Python 代码需要类型标注；
- 代码需要**尽最大可能保证精简**和正确，**严禁代码堆砌和冗余逻辑（例如不允许定义的数据结构、接口字段冗余，不允许代码出现不必要的条件判断和“错误处理”，满足功能就不需要添加了）**；
- 在保证上述要求的前提下，再考虑代码模块化，不同模拟器后端能够较为容易的控制管理；
- 必要时允许使用 C++ 代码（不必要时不要使用），注意代码构建管理；

## 实现说明
- 目前**已经实现了对 Gazebo、MuJoCo 模拟器后端的控制**。后续可能会接入 PyBullet、Habitat-Sim 等模拟器；
- 你需要仔细考虑添加需求后应该如何设计，例如是否需要变更接口和模拟器后端、`recorder_lerobot.py` 的原本实现、怎么变更才能保证架构精简和正确，等等；
- 为了帮助你实现，我在 `refs/lerobot/` 下存放了 lerobot v0.5.1（目前最新）包的源码和文档。与 lerobot dataset 数据集相关的文档在 `refs/lerobot/docs/source/` 下，建议你仔细阅读 lerobot 的文档，充分利用此库的能力；
- 如果你不清楚模拟器后端接口/gRPC Server 接口的含义，可以查看 `control_stubs/control_stubs/*.proto` 的注释内容，或者阅读实际的后端代码；
- 为了实现需求，你可以在保证 `代码规范和红线` 的前提下尽情修改，**包括上面提到的 gRPC 接口定义，以及后端模拟器（如果 gRPC 定义的数据结构等阻碍了你实现，可以果断修改）**，不需要考虑向前/向后的兼容性，但是需要兼容现有的 GazeboBackend、MuJoCoBackend 实现。
- 你对于这个框架的主要实现设计、架构设计应该记录在 `DESIGN.md`，防止遗忘导致架构混乱。可以不需要很长内容；
- 你的短期目标/里程碑应该记录在 `TODO.md` 中；
- 你需要对你的设计实现拟定相应的单元测试，不需要很多，保证接口的功能即可；


## 环境说明
- 本项目开发环境统一使用 miniforge 管理的虚拟环境。该环境已经存在，使用 `mamba activate robosim` 即可激活该虚拟环境（需要先 `eval "$(mamba shell hook --shell bash)"`），激活后使用 `mamba list` 可以查看已安装的包；
- robosim 虽然是 miniforge 环境，但已经提供了 ROS2 Humble 相关的 python 包，因为它使用了 RoboStack 项目；
- robosim 环境已经提供了固定 gRPC 的版本（`grpcio==1.78.1`,`protobuf==6.33.5`），你不得更改这个版本；
- robosim 环境已经提供了 MuJoCo python bindings 包（`mujoco`）；
- robosim 环境已经提供了 lerobot 0.5.1 版本的包（`lerobot`）；
- 使用 `./scripts/gen_protos.sh` 来编译 proto 文件生成 python 和 C++ stubs；
- 不得使用系统全局的 Python 环境、不得自行创建新环境。如有必要需要先说明。

## 验收标准
- 你所编写的单元测试通过；
- gRPC 的接口基本实现（除了被注释的 `ExecuteNaturalCommand`、`GetAgentStatus`）；
- 对于 MuJoCo 后端，你暂时不需要实现 StepPhysics 这个接口，默认情况需要不依靠用户调用就自动进行步进运算（除非用户调用了 pause 且没有调用 resume）；以后我会自行添加仿真暂停选项一并实现；
- 对于 MuJoCo 后端，`python3 -m robosim.server --backend mujoco` 至少需要正确启动；
- robosim 环境提供了 `ruff` 和 `mypy`。你需要创建 `pyproject.toml` 并通过 `ruff` 和 `mypy` 的 lint 检查；

