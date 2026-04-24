
## QUICKSTART

环境准备。以 miniforge 管理虚拟环境为例：

```bash
mamba env create -f environment.yml
mamba activate robosim
```

启动 GazeboBackend 需要额外启动 ROS2 节点（后续会集成进 `server.py`）。在新的窗口中使用 robosim 虚拟环境：

```bash
mamba activate robosim
pushd drivers_sim/gazebo-11/
# 解压 gazbo 预设模型
tar -zxpvf assets-model.tar.gz
# 构建 Gazebo 项目
colcon build
source ./install/setup.bash
popd
mamba activate robosim
ros2 launch demos gzsim.nav2.launch.py
```

再编译 proto 接口、启动 gRPC server：

```bash
mamba activate robosim
./scripts/gen_protos.sh --clean
./scripts/gen_protos.sh
python3 -m robosim.server --port 50051
```

## 开发规约与环境说明

- robosim 环境提供了 `ruff` 和 `mypy`。在 PR/提交前需要通过 `ruff` 和 `mypy` 的 lint 检查。之后我会设置 pre-commit hooks；
- 本项目开发环境统一使用 miniforge 管理的虚拟环境；
- robosim 环境已经提供了固定 gRPC 的版本（`grpcio==1.78.1`,`protobuf==6.33.5`），不得随意更改这个版本，这提供了对 OpenHarmony ArkUI 的兼容性；

## 资产规约

精简版简化了资源目录的放置方法。按照模拟器类型区分存放目录，`drivers_sim/gazebo-11/assets` 存放 Gazebo Classic 的资源，`drivers_sim/mujoco/assets` 存放 MuJoCo 的资源。

