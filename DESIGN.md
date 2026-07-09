# RoboSim 框架设计

## 目标
为多种模拟器后端（Gazebo/MuJoCo/PyBullet/Habitat-Sim）提供统一的控制和状态读取抽象，通过 gRPC 接口向上层暴露。

在 thesis-level benchmark generator 中，`vsim` 还承担 Concrete Scenario
Definition（CSD）的后端实现边界：上层 benchmark generator 负责从自然语言
benchmark distribution 采样并固化一个具体 CSD；`vsim` 负责将该 CSD
finalize/load 为 MuJoCo、Gazebo 或未来后端所需的 native scene/artifacts，
并继续提供渲染、传感器、控制、policy runtime 和 rollout 采集能力。CSD 在传入
`vsim` 时已经不是 distribution，而是一个具体 task instance 的固定定义。

后端 native scene 生成不应放在 thesis-level benchmark generator 中。若在
`vsim` 内实现 MuJoCo/Gazebo 等后端的 CSD realization，必须先仔细阅读对应
官方文档；例如 MuJoCo 路径需要理解 MJCF 的 body、asset、geom、inertial、
joint、material、mesh、contact 参数以及 compiler defaults 等语义。

同一个 CSD 可以 realization 到多个 backend，并缓存各自的 native artifacts
和 backend manifest。缓存 key 至少应包含 CSD 内容 hash、所选 asset IDs 与
backend adapter/resource hashes、目标 backend、realization config、`vsim` realization
版本、可获取的 simulator 版本以及 sampled randomization values。CSD 始终是
场景语义源，MJCF/URDF/SDF/Gazebo 资源等只是可复现的派生产物。
MuJoCo realization 若发现 existing `manifest.json` cache key 匹配且 manifest
列出的 generated/preview files 均存在，会直接返回 cached manifest，避免再次依赖
原始 asset cache 或临时 robot template source。

CSD realization 需要显式处理 asset backend compatibility。不同 backend 对
mesh 格式、material/texture、collision geometry、articulation/joint、sensor、
lighting、scale、frame/up-axis、contact 参数和 inertial 语义的支持可能不同；
不能直接复用或只能有损转换时，应生成 validation failure 或 blocker。
数值字段也属于 backend 语义的一部分：单位、mesh scale、坐标系/up-axis、
inertial 默认值、friction/contact 参数和 collision geometry 的解释都必须按目标
模拟器官方文档实现，不能从另一个 backend 的写法直接照搬。

官方文档起始参考与当前 realization 依据：
- MuJoCo MJCF XML Reference:
  `https://mujoco.readthedocs.io/en/stable/XMLreference.html`
- ROS URDF XML documentation: `https://wiki.ros.org/urdf/XML`
- SDFormat specification: `https://sdformat.org/spec/`
- Gazebo Sim resource lookup:
  `https://gazebosim.org/api/sim/8/resources.html`
- Gazebo Sim SDF/resource migration notes:
  `https://gazebosim.org/api/sim/8/migrationsdf.html`
- OpenUSD asset resolution and composition references:
  `https://openusd.org/dev/api/ar_page_front.html`,
  `https://openusd.org/release/glossary.html`

当前 CSD realization 的已实现范围包括：

- 后端输入 gate：`vsim` 可以检查一个 CSD 引用的 asset 是否具备目标 backend
  resource adapter，并提取参与 cache key 的 adapter/resource hashes。若缺失
  adapter，会返回 typed blocker。发布到 asset library 的记录应已经通过
  catalog 质量门禁，失败或待修复候选不应出现在这里。
- 第一版 CSD compiler：`compile_csd(..., backend=...)` 将固定 CSD 和 asset
  registry 编译到 benchmark package 的
  `engine_manifests/<backend>/<csd_id>/...` backend slot，并返回
  `CsdRealizationManifest`。调用 API 时应把 `output_root` 传为 package 下的
  `engine_manifests/`。当前 MuJoCo 路径生成 `scene.xml`，Gazebo 路径生成
  `world.sdf`。该路径目前支持具备对应 backend
  resource adapter 的刚体 mesh 对象、CSD 显式 pose、MuJoCo 动态对象
  `freejoint`、MuJoCo visual mesh 与可选 collision mesh 分离、Gazebo SDF
  model/link/visual/collision、mass/friction 标量，以及 realization cache key。
  它不替代后续 runtime load/render/physics validation。
- 后端目标入口：协作者应优先调用 `compile_csd(..., backend=...)`，而不是把
  某个 backend 编译器当作唯一抽象。当前 `backend="mujoco"` 与
  `backend="gazebo"` 已实现第一版文件生成。ROS2 launch/package/share 目录
  属于后续 runtime integration 约定，不是 Gazebo SDF 编译产物的前置条件。

### CSD realization 输出布局

`vsim` 的 compiler 输出必须落在 thesis benchmark package 的统一 backend slot
中，而不是写到 backend 自己的临时目录或 `drivers_sim` 源目录：

```text
engine_manifests/
  <backend>/
    <csd_id>/
      manifest.json
      <backend-entry-file>
      assets/
        ...
      diagnostics/
        ...
```

调用方应把 `compile_csd(..., output_root=...)` 的 `output_root` 设为 benchmark
package 下的 `engine_manifests/`。MuJoCo 目标写入
`engine_manifests/mujoco/<csd_id>/scene.xml`；Gazebo 目标写入
`engine_manifests/gazebo/<csd_id>/world.sdf`。`manifest.json` 持久化
`CsdRealizationManifest`，并记录 cache key、entry file、generated files、
copied dependency files、backend、simulator/compiler version 和后续 preview
files。

该布局借鉴 OpenUSD 对 asset identity、composition 与 resolved asset path 的
分离，但 MVP 不要求生成 USD 文件。CSD 和 asset registry 持有 project-owned
asset IDs、语义、provenance 和 backend resource adapters；MuJoCo/Gazebo
compiler 负责把这些逻辑 asset 解析为当前 realization 目录中的相对路径。CSD
中的 environment、robot、objects、enum relationships、材质、collision 与
domain-randomization override 应作为概念上分层的输入处理，即使最终需要为
MuJoCo 生成一个可加载的 MJCF entry file。

MuJoCo realization 的路径规则：

- `scene.xml` 必须能从 `engine_manifests/mujoco/<csd_id>/scene.xml` 直接被
  `mujoco.MjModel.from_xml_path()` 加载；
- MJCF 中的 mesh、texture、include 等依赖不得指向原始下载 cache 或
  `drivers_sim` 源目录；
- CSD object backend resources 必须复制到
  `engine_manifests/mujoco/<csd_id>/assets/...` 后再被 MJCF 引用；
- 临时复用 `drivers_sim/mujoco/assets/robots/...` 中的 robot/world 模板是允许的，
  但 compiler 必须复制所需 XML、mesh、texture、SRDF/metadata 等 dependency
  closure 到当前 realization 目录，不能把源目录当作永久运行时依赖；
- 需要依据 MuJoCo XML Reference 使用 `compiler meshdir`、`texturedir` 或
  `assetdir` 等机制保证 entry file 内路径清晰、相对、可移动；
- compiler 可以先支持 Franka/tabletop manipulation MVP，但接口和目录结构不得把
  Franka、tabletop 或 MuJoCo 写死为唯一目标。

实现记录（2026-07-08）：MuJoCo 路径设计前已查阅 MuJoCo MJCF XML Reference
中关于 `asset/mesh`、`geom` mesh 引用、mesh scale、mesh frame centering、
collision convex hull、material 与 contact/friction 参数的说明。由此确认第一
阶段不能把 project asset ID 直接当作 MJCF 文件路径使用，必须先通过 asset
backend adapter/compatibility gate，再进入后续 MJCF 生成。

实现记录（2026-07-09）：当前 `compile_csd_to_mujoco()` 继续依据 MuJoCo
MJCF XML Reference（stable）中的 `compiler`、`asset/mesh`、`worldbody/body`、
`include`、`freejoint`、`geom`、mesh `file`、mesh `scale`、`texture`、
`material`、geom `mass` 与 `friction`
语义实现。无机器人模板时，编译器使用 `<compiler meshdir="...">` 指向编译产物
目录内的 `assets/`，并设置 `texturedir` 让 texture 也从 backend-local assets
解析；有机器人模板时，顶层 `scene.xml` 通过相对 `<include>` 引用 realization
package 内复制出的 Franka MJCF 模板，并沿用模板内的 `compiler meshdir` 规则。
CSD object 使用 `<asset><mesh file="relative/path.obj"/></asset>` 注册 visual
mesh resource，可带 mesh `scale`；若 backend resource adapter 提供
`collision_mesh_path`，compiler 会注册单独的 collision mesh，并把 visual geom
设为 `contype="0"`、`conaffinity="0"`，由透明 collision geom 承载 MuJoCo
collision/mass/friction。object `initial_state` 解析为 typed physical state，
当前支持 `mass_kg` 与 `friction`，并由 compiler 显式写入对应 object geom。
MuJoCo geom `friction` 按 MJCF `real(3)` 输出；CSD scalar friction 被解释为
sliding friction，并保留 MuJoCo 默认 torsional/rolling friction
`0.005 0.0001`，CSD 3-vector friction 则原样写出。
adapter material/texture metadata 会生成 MJCF `texture`、`material`，再由 object
visual geom 引用。动态对象以 `freejoint` 表示自由刚体。无 robot template 的
MuJoCo scene 会把 CSD `environment.gravity` 写入 `<option gravity="...">`；带
robot include 的 scene 会 patch realization package 内复制出的 robot template
entry XML 的 `<option gravity="...">`，不修改源 template，也不在顶层 scene 中
额外生成冲突的 `<option>` 节点。当前支持的 world template 为 `empty_floor` 和
`world_tabletop`；
`world_tabletop` 会生成 backend-local static tabletop geometry，而不是引用
`drivers_sim` 的 world scene。编译器写出 MJCF 后会立即用
`mujoco.MjModel.from_xml_path` 做 package-local load check，并在
`diagnostics/load_check.json` 记录 `model_load`、gravity、CSD object body pose
和 environment surface pose 检查结果；若该检查失败，compiler 返回
`CsdRealizationBlocker(scope="vsim_realization")`，不发布 manifest。
load check 通过后，compiler 会写入 `diagnostics/relationship_check.json`，再运行
短 MuJoCo forward/step stability check，写入 `diagnostics/physics_check.json`；
该检查只证明基本数值稳定，不代表 task success 或 rollout 质量。随后 compiler
会使用第一个 CSD camera 做 MuJoCo offscreen render，写入
`diagnostics/semantic_preview.ppm`，并在 `manifest.preview_files` 中记录该 preview
artifact；若 relationship/physics check、渲染或输出为空失败，同样返回 blocker。

MuJoCo compiler 会在写出文件前执行语义 gate，避免生成可加载但语义错误的
MJCF。当前会阻止非 `units="m"`、非 `frame="world"`、以及非 `box` 类型的
environment surface。被阻止的情况返回 typed
`CsdRealizationBlocker(scope="csd")`，调用方应修复 CSD 或等待 compiler 增加
明确转换策略，而不是依赖静默降级。
compiler 还会检查 enum relationship 的 `subject` 和 `object` entity refs 是否
能解析到当前 CSD 中的 object、environment surface 或 robot；无法解析的
relationship 会作为 CSD blocker 返回。当前 MuJoCo relationship diagnostics
还会对 `avoid_contact` 做确定性数值检查：读取关系中的 `min_distance_m`，比较
已加载 MuJoCo body 的初始位置距离，失败时写入
`diagnostics/relationship_check.json` 并返回
`CsdRealizationBlocker(scope="csd")`。当前不在 compiler 中推断 `inside`、
`on_top_of` 等关系的几何成立性，因为这需要 asset extents、support/contact
semantics 和 evaluator predicate 共同定义；在这些契约补齐前，compiler 不做
自然语言或 AI 判定。
MuJoCo asset compatibility gate 还会阻止不受 MJCF mesh asset 支持的 mesh
resource 扩展名；当前允许 `.obj`、`.stl`、`.msh`，visual mesh 与 collision
mesh 都适用。此检查发生在复制资源和写出 MJCF 之前，返回
`CsdRealizationBlocker(scope="asset")`。

编译产物目录必须自包含当前 backend 需要的资产文件。MuJoCo 编译器会把
CSD 引用的 backend mesh resources 复制到
`engine_manifests/mujoco/<csd_id>/assets/<resource-relative-path>`，`scene.xml`
只引用该目录内的相对路径。当前 MuJoCo compiler 会把该产物提升为
`engine_manifests/mujoco/<csd_id>/` 下的完整 realization package，持久化
`manifest.json`，创建 `diagnostics/`，写入 `diagnostics/load_check.json` 与
`diagnostics/relationship_check.json`、`diagnostics/physics_check.json`、
`diagnostics/semantic_preview.ppm`，并把临时 Franka robot template 与其 mesh
dependency closure 复制到当前 realization package。这样即使原始 asset cache 或
`drivers_sim` 源目录移动或清理，已编译的 MJCF 仍可加载。后续处理 OBJ 材质、
纹理、URDF/SDF resource 时也必须遵守同样原则：native scene artifact 不得依赖
易丢失的下载缓存路径。

MuJoCo runtime loading 通过 `MuJoCoBackend.from_csd_realization_manifest()` 或
`MuJoCoBackend.from_csd_realization_manifest_file()` 消费 compiler 输出的
`CsdRealizationManifest`。文件入口以 `manifest.json` 所在目录作为 package root
解析 `entry_file`，避免依赖可能过期的绝对 `root_path`。解析后的 `scene.xml`
仍交给现有 `MuJoCoBackend(scene_path=...)` 初始化路径处理，避免复制或绕过现有
servo、sensing、recording、policy runtime 行为。
server 启动路径也可使用 `python -m robosim.server --backend mujoco
--csd-manifest <engine_manifests/mujoco/<csd_id>/manifest.json>` 直接加载 compiled
CSD realization；普通 `--scene` 路径保持可用。

CSD compiler tests must keep scenario definitions as JSON fixtures under
`tests/fixtures/csd/` instead of embedding large dictionaries in test code.
Fixtures must use the structured CSD scenario contract: `environment`, `robot`,
`objects`, enum-typed `relationships`, sampled overrides, sensors, and
evaluator refs. Fixture coverage should include robot tabletop scenes, multiple
static/dynamic objects, object-only scenes, unsupported robot blockers, and
material/texture/scale adapter resources. At least one MuJoCo compiler smoke
test should load the compiled `scene.xml`, render an offscreen screenshot from
`world_camera` into `diagnostics/`, and assert basic CSD semantic preservation
such as object presence, pose sanity, and nonblank rendered pixels.
MuJoCo fixture coverage must also include an adapter with a separate collision
mesh resource so visual geometry, collision geometry, copied dependencies, and
loadability are tested together.

Implementation code must not treat CSD JSON as an unstructured `dict[str, Any]`
after the API boundary. JSON fixtures and package artifacts are parsed into the
typed `ConcreteScenarioDefinition` dataclass model in `robosim.core.csd`,
including typed environment, robot, object, pose, camera, light, surface, and
enum relationship records. Object physical state is represented by
`CsdObjectInitialState`, not by ad hoc map access. Backend compiler code should
consume those typed objects so collaborator mistakes in key names, relationship
types, or shape conversions fail at parse time instead of silently changing
scene semantics.
Backend resource adapters are also typed at the compiler boundary. The
`BackendResourceAdapter` model carries the project asset ID, backend, resource
hash, visual mesh path, optional mesh scale, optional material/texture, and
optional collision mesh path. Compiler internals should use this model instead
of anonymous registry dictionaries.

实现记录（2026-07-09）：第一版 `compile_csd_to_gazebo()` 依据 SDFormat
1.12 的 `sdf/world/model/link/visual/collision/geometry/mesh/uri` 结构与 Gazebo
Sim resource lookup 文档实现。Gazebo 产物写入
`engine_manifests/gazebo/<csd_id>/world.sdf`，并复制 CSD 引用的 Gazebo backend
resources 到 `engine_manifests/gazebo/<csd_id>/assets/...`。SDF 内 mesh URI
使用相对路径 `assets/...`，使 `world.sdf` 可随 artifact root 移动；不要求生成
ROS2 package、launch 目录或安装到 package share。后续 runtime 加载时可通过
当前工作目录、绝对路径或 `GZ_SIM_RESOURCE_PATH` 暴露 artifact root，但编译器
本身只负责生成可审计、可缓存、资产自包含的 backend-native 文件。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      gRPC Server                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │SimulationSvc│ │ SensingSvc  │ │RobotCoreSvc │ │Mobility│ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───┬────┘ │
└─────────┼───────────────┼───────────────┼────────────┼──────┘
          │               │               │            │
┌─────────▼───────────────▼───────────────▼────────────▼──────┐
│                   Backend Manager                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SimulatorBackend (Abstract Base)           │   │
│  │  + get_robot_state()                                 │   │
│  │  + get_robot_spec()                                  │   │
│  │  + set_joint_target()                                │   │
│  │  + get_end_effector_state()                          │   │
│  │  + list_sensors() / get_sensors() / stream_sensors() │   │
│  │  + navigate_to() / get_robot_pose_in_map()           │   │
│  │  + reset_world() / step_physics()                    │   │
│  │  + emergency_stop()                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              GazeboBackend (Concrete)                       │
│  - 使用 ROS2 topic 动态发现传感器                              │
│  - 订阅 JointState, Imu, LaserScan, Image 等                 │
│  - 发布 /cmd_vel 控制                                         │
│  - 使用 actionlib 调用 Nav2                                   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│              MuJoCoBackend (Concrete)                       │
│  - 使用 MuJoCo Python Bindings 进行仿真                       │
│  - 从 XML 模型文件加载机器人配置                                │
│  - 支持关节位置/速度/扭矩控制                                   │
│  - 通过 forward kinematics 获取末端执行器位姿                   │
│  - 支持力/扭矩传感器、IMU 等                                    │
│  - 默认启用重力补偿                                            │
│  - 非 headless 模式支持被动 viewer                             │
└─────────────────────────────────────────────────────────────┘
```

## 模块结构

包含部分文件，仅作示意。

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

## 设计细节

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

## 关于 MuJoCoBackend

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
- MuJoCo 的 offscreen renderer 绑定创建它的线程；后端按“线程 + 分辨率”缓存 renderer，避免录制线程和 gRPC 请求线程跨线程复用 EGL/OpenGL 上下文导致黑帧/花屏；
- 当前 gRPC 扩充了力/力矩传感器类型与数据结构，以避免 MuJoCo 现有传感器语义丢失。

## 关于 LeRobot 数据录制

### 1. 录制入口
- `RobotDataService`，仅暴露 `EpisodeStart` / `EpisodeEnd`；
- gRPC server 启动时构造一个 `LerobotDataRecorder`，数据根目录固定为仓库下 `data/lerobot/<repo_name>`；
- 同一 backend 实例同一时刻只允许一个录制 session。

### 2. 采样模型
- `EpisodeStart` 读取一次 backend 的 `robot_state`、`robot_spec`、`end_effector_state`、`sensor snapshot`，先推导数据集 schema，再立即写入第一帧；
- 后续由后台线程按 `fps` 继续采样；
- `EpisodeEnd` 停止采样线程、保存 episode，并立即 `finalize()`，保证每次结束后磁盘上的数据集都是可加载状态。

### 3. 字段映射
- 关节状态统一映射为：
  - `observation.state`
  - `observation.velocity`
  - `observation.effort`
- `action` 固定定义为 joint coordinate system 下的绝对关节位置目标，记录 backend 最近一次 joint command 的 position target，保证 replay 与 policy runtime 共享同一动作语义；
- 末端位姿映射为 `observation.end_effectors.<group>.{position,orientation}`；
- 视觉数据映射为 `observation.images.<sensor>`；
- IMU / LiDAR / Odom / Force / Torque 各自映射到对应 `observation.*` 数值向量。

### 4. 过滤规则
- `jmg_excluded` 先于 `jmg_included` 生效；
- 未显式指定 `jmg_*` 时，默认录制当前 `get_robot_state()` 返回的全部关节；
- 未显式指定 `sensor_*` 时，默认录制全部非 joint 传感器；
- 现阶段不重复把 joint sensor 写入 dataset，因为它和 `observation.state` 语义重叠。

### 5. LeRobot 落盘策略
- 直接复用 `lerobot>=0.5` 提供的 `LeRobotDataset.create()/resume()/add_frame()/save_episode()/finalize()`；
- 当前相机帧使用 `image` 特征直接写盘，不走 mp4 编码，先保证 v3 数据结构稳定和测试确定性；
- `lerobot` 会在写 parquet 时把图片字节嵌入 parquet，再删除 `images/<camera>/episode-*` 下的临时 PNG；recorder 在 episode 结束后继续清理残留空目录，避免留下误导性的空相机目录；
- 已存在 repo 继续录制时使用 `resume()`，并校验 `fps` 与 feature schema 必须完全一致。

### 6. 数据集重放
- `RobotDataService` 额外暴露 `EpisodeReplay(repo_name, episode_id)`，语义是阻塞式单 episode 回放；
- 回放时从本地 `data/lerobot/<repo_name>` 打开 LeRobotDataset，并通过 `episodes=[episode_id]` 只读取目标 episode；
- 回放只消费 dataset 的 `action` 向量，固定以 `JointCommand.POSITION` 下发到 backend，因此 replay 语义与 LeRobot 常见 joint-space absolute action 一致；
- 为避免 `all` / 子 group 同时包含相同 joint 集合导致 MuJoCo `set_joint_target()` 组选择歧义，回放前先读取当前 `RobotSpecification`，优先选择 joint 列表精确匹配的最小 joint model group；
- Gazebo 当前不实现回放写入路径，相关方法允许直接 `NotImplementedError`。

## 关于 LeRobot IL 推理支持

### 1. 当前范围
- 当前阶段只支持 MuJoCo 后端上的 LeRobot IL policy 推理；
- 不实现训练支持；
- 首个目标是兼容 ACT 这类标准 joint-space chunking policy，但运行时适配层应尽量保持对其他 LeRobot IL policy 通用。

### 2. 录制 / 回放 / 推理
- replay 继续消费 dataset 中的 `action`，并按 joint-space absolute position target 下发；
- 推理期的 observation 不直接复用 recorder 的“全量 feature schema”作为 policy 输入；
- 推理期只暴露最小必需 observation：`observation.state`、`observation.images.*`、`task`；
- recorder 允许继续记录更丰富的 feature，但这不应反向决定 policy runtime 的输入结构。

### 3. action 语义
- 框架内 LeRobot policy runtime 的标准 action 语义固定为 joint-space absolute position target；
- recorder 的 `action` 应记录“可 replay 的绝对关节目标”；
- backend 若内部直接运行 POSITION 控制，则 `action` 可直接取最近一次 joint command；
- backend 若当前控制源是 VELOCITY / TORQUE / twist 等非绝对位置命令，则必须先归一化成绝对关节位置目标再暴露给 recorder；至少不能把这类原始命令直接写进 `action`，否则 replay 按 POSITION 下发时会失真。

### 4. 运行时结构
- server 内独立的 policy runner，职责仅限于：
  - 加载 checkpoint / preprocessor / postprocessor / policy；
  - 维护推理线程与 stop/reset 状态；
  - 在固定控制频率下执行 observation -> preprocess -> `select_action()` -> postprocess -> `set_joint_target()`；
  - 在开始新一轮推理或 world reset 后调用 `policy.reset()`，清理 LeRobot policy 内部的 action queue / history。
- policy runner 与 recorder / replay 必须互斥，同一 backend 实例任一时刻只能存在一种主动控制源。

### 5. Observation 适配
- backend snapshot -> LeRobot observation 的转换逻辑应从 recorder 中抽离成共享组件；
- 该适配层负责：
  - 选择推理需要的 joints / cameras；
  - 生成与 policy checkpoint 对齐的 observation keys；
  - 维持 joint names 与 `JointModelGroup` 的稳定映射；
  - 在 postprocess 后把 action 向量重新映射为 backend 可执行的 joint target。

## 关于 ServoControlStream 调试客户端

- 调试客户端位于 `control_stubs/tools/servo_keyboard.py`，职责仅限于把终端键盘事件转成 `ServoCommand` 流，不引入新的控制抽象；
- 客户端启动时先读取 `RobotSpecification`，默认自动选择一个带 end effector 的 `jmg` 作为 twist 控制目标；若存在较小的非 ee `jmg`，则同时把它作为 joint 调试目标；
- 终端 raw mode 只能稳定拿到 key-down，不能可靠拿到 key-up，因此客户端采用“按键生效一小段保持时间，超时后自动补发零速度”的语义，避免后端持续保持旧 velocity target；
- `ServoCommand` 是 `oneof`，因此 twist 和 joint 调试命令始终分开发送，客户端每个周期最多发一条 twist 命令和一条 joint 命令。
