# Habitat-Sim GPU 运行与图像查看指南

这份指南用于在有 GPU/EGL 的机器上启动 RoboSim 的 Habitat-Sim 后端，并通过
`habitat_rgb` gRPC camera 查看图像。

## 1. 准备 Python proto 文件

如果 `control_stubs` 里缺少 `*_pb2.py`、`*_pb2_grpc.py` 或 `*.pyi`，server/client 都会
无法导入 gRPC 类型。先在仓库根目录运行：

```bash
bash scripts/gen_protos.sh
```

这个脚本会在 `control_stubs/control_stubs/` 生成 Python/C++ gRPC 文件，并把 Python
文件和 `*.pyi` 拷贝到 `control_stubs/` 根目录，方便 `from control_stubs import ...`
正常工作。

如果想清掉后重新生成：

```bash
bash scripts/gen_protos.sh --clean
bash scripts/gen_protos.sh
```

## 2. 启动 Habitat + Panda + Camera

如果资产目录在仓库外，例如 `/home/murphy/code/drivers_sim`，设置
`ROBOSIM_DRIVERS_SIM_ROOT`：

```bash
ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
python -m robosim.server \
  --backend habitat \
  --robot panda \
  --scene habitat/assets/worlds/apartment.glb \
  --habitat-enable-camera \
  --headless \
  --port 50051
```

说明：

- `--robot panda` 会加载 Panda URDF articulated object。
- `--scene habitat/assets/worlds/apartment.glb` 会加载 Habitat 环境场景。
- `--habitat-enable-camera` 会创建 `habitat_rgb` camera renderer，需要 GPU/EGL。
- 这里要用 `--headless`，不要用 `--no-headless`。`--no-headless` 走 Habitat 官方 viewer
  subprocess，只能直接打开 `.glb/.obj` 场景，不能动态加载 Panda URDF。

如果只想加载 Panda，不加载环境场景，可以去掉 `--scene`：

```bash
ROBOSIM_DRIVERS_SIM_ROOT=/home/murphy/code/drivers_sim \
python -m robosim.server \
  --backend habitat \
  --robot panda \
  --habitat-enable-camera \
  --headless \
  --port 50051
```

## 3. 保存一帧 `habitat_rgb`

另开一个终端，在同一个环境和仓库根目录下运行：

```bash
python - <<'PY'
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

## 4. 连续显示 `habitat_rgb`

如果安装了 OpenCV，可以连续显示图像流：

```bash
python - <<'PY'
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

## 常见问题

- `ModuleNotFoundError: No module named 'control_stubs.common_pb2'`：
  先运行 `bash scripts/gen_protos.sh`。
- `Habitat-Sim viewer cannot load MuJoCo MJCF XML scenes`：
  `--scene` 不能传 MuJoCo `scene.xml`。Habitat scene 要使用 `.glb`、`.gltf`、`.obj`
  或 `.ply`。
- `--no-headless` 看不到 Panda：
  这是预期行为。Panda 是通过 backend 的 Simulator API 动态加载的，只能走
  `--headless --habitat-enable-camera` 后通过 gRPC camera 查看。
