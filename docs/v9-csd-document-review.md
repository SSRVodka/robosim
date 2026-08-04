# v9 CSD 文档变更审阅稿

本稿集中列出 `main..feat/hierarchy-compiler` 中 `README.md`、`DESIGN.md`
与 `TODO.md` 的新增内容，供审阅使用；不改变三份原文档的职责。

## README：用户入口

新增 v9 MuJoCo resource package 的最短编译命令：

```bash
MUJOCO_GL=egl python -m robosim.compile \
  --csd example/art_6b18395c757141bb9fa08cbcb7e6bc87/scene.usda \
  --output-root /tmp/robosim-engine-manifests
```

该命令输出 realization manifest，在
`<output-root>/mujoco/<scene_id>/` 写入 `scene.xml`、`models/` 与 diagnostics；
被拒绝时打印 typed blockers 并以状态码 2 退出。

新增不启动 gRPC server 的 native scene 查看命令：

```bash
python -m robosim.view --backend mujoco --entry /path/to/scene.xml
python -m robosim.view --backend pybullet --entry /path/to/scene.py
python -m robosim.view --backend gazebo --entry /path/to/world.sdf
```

来源：[README.md](../README.md)。

## DESIGN：架构事实

新增的 v9 realization 设计说明包括：

- 输入是 `scene-export/v9-vsim-articulated-resources` package；`scene.usda`、
  `manifest.json`、`checksums.sha256` 与 referenced `asset.usda` 是唯一资源来源。
- reader 使用 `Usd.Stage.Open(..., LoadAll)` 读取 composed stage，验证 package
  boundary、checksum、米制/Z-up、单位 instance scale；composed prim path 是 instance
  identity，composed `assetserver:asset:id` 是 asset identity。
- 唯一 `(asset_id, resource_digest)` realization 为一个 package-local MJCF submodel；
  scene 用 `<frame><attach>` 实例化。runtime 不能依赖 `drivers_sim` 或下载缓存。
- `/World/Robot` 是 compiler-owned fixed-base template descriptor；复制 XML、SRDF、
  mesh closure 后 patch pose。模板 closure hash 属于 cache key，并保留 actuator/SRDF
  control metadata。
- `/World/Cameras` 和 `/World/Lights` 的 authored Camera/DistantLight 映射为 native
  camera/light；无 authored camera 时才创建 preview fallback `world_camera`。
- articulation initial state 只读 composed `PhysicsDriveAPI:*:physics:targetPosition`，
  写入 `runtime/initial_joint_positions.json`，由 compiler diagnostics 和 backend 在
  首次 forward 前应用。
- visual material binding、texture closure、joint frame 变换均由 composed USD 读取；
  child body zero pose 是 `localFrame0 × inverse(localFrame1)`。

来源：[DESIGN.md](../DESIGN.md)。

## TODO：里程碑状态

新增 `Checkpoint 6: MuJoCo v9 articulated resource packages（进行中）`：

- 目标：用 v9 package reader 取代 registry input；realize static/rigid/articulated
  assets；验证 production package 的 load/step/render/relocatability；保留 composed
  material/texture/light 与正确 joint-frame kinematics。
- 已完成：读取 `/World/Robot` 并生成 package-local MJCF/SRDF template；保留 authored
  camera 与 distant light，并用 `example/experiment` 验证。
- 待完成：补 composed-layer 与 drive-initial-state coverage，并删除不可达的 registry
  compiler body。

来源：[TODO.md](../TODO.md)。
