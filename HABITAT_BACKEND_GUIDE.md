# Habitat-Sim 后端启动指南

Habitat-Sim 后端和 Gazebo / MuJoCo 的启动路径差异较大：它依赖可选的
`habitat_sim` Python 包，显示窗口、headless camera renderer、Panda articulated object
分别走不同的 Habitat-Sim 运行模式。本指南集中说明 Habitat 后端的安装、启动和图像查看。

## 1. 准备 RoboSim 环境

先完成 README 中的通用环境创建和 proto 生成：

```bash
mamba env create -f environment.yml
mamba activate robosim
./scripts/gen_protos.sh --clean
./scripts/gen_protos.sh
```

如果 `control_stubs` 里缺少 `*_pb2.py`、`*_pb2_grpc.py` 或 `*.pyi`，
server/client 会无法导入 gRPC 类型，重新运行上面的 proto 生成命令即可。

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

不传入 `--robot-name` 时，Habitat 后端提供渲染能力，不控制机器人：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --scene <your-scene.glb>
```

`--scene` 需要传 Habitat-Sim 支持的 mesh / scene 文件，例如 `.glb`、`.gltf`、
`.obj` 或 `.ply`。

## 4. 本地显示窗口和软件渲染

在没有 NVIDIA GPU 但有本机显示器的环境中，可以使用普通显示版 Habitat-Sim 和 Mesa
软件渲染：

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=4.1 DISPLAY=:0 \
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --scene drivers_sim/mujoco/assets/worlds/two_bedroom_apartment/BEDROOM_NEO/model.obj \
  --no-headless
```

`--no-headless` 会打开 Habitat-Sim viewer 窗口。这个模式只适合直接查看单个 mesh 场景，
不会通过 gRPC 暴露 `habitat_rgb` 图像，也不能动态加载 Panda URDF。

## 5. 外部 drivers_sim 资产

如果 `drivers_sim` 资产目录不在 robosim 仓库内，可以用
`ROBOSIM_DRIVERS_SIM_ROOT` 指向外部资产根目录。`--scene` 支持相对该目录的路径：

```bash
ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --scene habitat/assets/worlds/apartment.glb \
  --no-headless
```

## 6. 加载 Panda 机器人

可以用 Habitat-Sim 的 articulated object API 加载 Panda 机器人：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot-name panda \
  --headless
```

Panda 模式会加载仓库中的 Panda URDF，并暴露 `panda_arm`、`panda_hand`、
`panda_arm_hand` 三个 joint model group。为了能在无 GPU 环境中运行，Panda 模式默认关闭
Habitat camera renderer，只提供 joint state/spec 和 POSITION joint target。

在有 GPU/EGL 的机器上，可以显式打开 camera renderer，并通过 sensing gRPC 接口读取
`habitat_rgb`：

```bash
ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot-name panda \
  --scene habitat/assets/worlds/apartment.glb \
  --habitat-enable-camera \
  --headless
```

Panda 可视化走 backend 的 camera renderer，不走 `--no-headless` 的 Habitat viewer
subprocess。后者只能直接打开单个 mesh 场景，不能动态加载 Panda URDF。

如果只想加载 Panda，不加载环境场景，可以去掉 `--scene`：

```bash
python3 -m robosim.server \
  --port 50051 \
  --backend habitat \
  --robot-name panda \
  --habitat-enable-camera \
  --headless
```

## 7. 保存一帧 habitat_rgb

启动带 camera renderer 的 Habitat 后端后，另开一个终端，在同一个环境和仓库根目录下运行：

```bash
python3 - <<'PY'
import numpy as np
from PIL import Image

from control_stubs.tools.client import RobosimClient

client = RobosimClient("localhost", 50051)
print(client.sensing.list_sensors())

data = client.sensing.get_sensors(["habitat_rgb"])
if not data.images:
    raise RuntimeError("No habitat_rgb image returned")

img = data.images[0]
arr = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
Image.fromarray(arr, "RGB").save("habitat_rgb.png")

print("saved habitat_rgb.png", img.width, img.height, img.encoding)
client.close()
PY
```

生成的 `habitat_rgb.png` 会保存在当前目录。

## 8. 连续显示 habitat_rgb

如果安装了 OpenCV，可以连续显示图像流：

```bash
python3 - <<'PY'
import cv2
import numpy as np

from control_stubs.tools.client import RobosimClient

client = RobosimClient("localhost", 50051)

for data in client.sensing.stream_sensors(["habitat_rgb"]):
    if not data.images:
        continue

    img = data.images[0]
    rgb = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("habitat_rgb", bgr)

    if cv2.waitKey(1) == 27:
        break

client.close()
cv2.destroyAllWindows()
PY
```

按 `Esc` 退出窗口。

也可以直接使用仓库中的小工具：

```bash
python3 view_habitat_rgb.py --help
python3 move_habitat_camera.py --help
```

## 常见问题

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
  `--headless --habitat-enable-camera` 后通过 gRPC camera 查看。
- `--headless` 读取不到 `habitat_rgb`：
  headless camera renderer 需要可用 EGL/GPU 渲染上下文。无 GPU 的本地显示环境可以改用
  `--no-headless` 直接打开 viewer，但该模式不提供 gRPC camera 图像。
