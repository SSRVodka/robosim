# MuJoCo OpenUSD Asset/Scene Compiler 设计

## 1. 范围与状态

本文定义 `vsim` 将 OpenUSD CSD realization 为 MuJoCo package 的下一迭代设计：

```text
asset.usd → asset.xml
scene.usd → scene.xml
```

本文的 v9 articulated-resource 迭代正在实现。输入是
`scene-export/v9-vsim-articulated-resources` package 的 `scene.usda`、同级
`manifest.json`、`checksums.sha256` 及 referenced `asset.usda`；不保留外部 JSON
asset registry 或 `asset_root` 兼容路径。实现期间遵循本文、`DESIGN.md` 和 `TODO.md`
已锁定的边界。

本轮修复通过 composed `Usd.Stage` 读取 visual material binding、texture dependency
closure 与 standard light/camera。articulation 的 child body zero transform 是
`localFrame0 × inverse(localFrame1)`；`localPos1` 仅用作 child-local joint position，
axis 为 `localRot1` 旋转后的 USD axis。

实现记录（2026-08-03）：child body translation 必须用上述完整 relative rotation
旋转 `localPos1`；不能只用 `localRot0`。该错误会让 `localFrame0 == localFrame1`
的关节仍产生错误 child offset。visual OBJ 的每个 `o` block 也必须单独注册为
MuJoCo mesh，因为 MuJoCo 只导入单一 OBJ 中的第一个 object。realization version
升至 `csd-compiler-0.6`。

CSD 与 AssetServer 输入契约见
[`openusd-csd-asset-contract.md`](./openusd-csd-asset-contract.md)。

## 2. 依据与版本

设计针对项目固定的 OpenUSD 26.05 和可获取的 MuJoCo 3.9.x，依据：

- MuJoCo MJCF XML Reference：
  <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- MuJoCo model editing/attachment：
  <https://mujoco.readthedocs.io/en/stable/programming/modeledit.html>
- MuJoCo OpenUSD importing（仅作为已评估的 native decoder 路径参考）：
  <https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html>
- OpenUSD stage/composition：
  <https://openusd.org/release/api/class_usd_stage.html>
- OpenUSD asset resolution：
  <https://openusd.org/release/api/ar_page_front.html>

MuJoCo `asset/model` 可加载 MJCF sub-model；`body/attach` 可把 child model body
subtree 复制到父模型并用 prefix 解决命名冲突。传统 `include` 是 XML DOM 插入，
且同一文件在一个模型中最多 include 一次，不作为重复资产实例化方案。

## 3. 目标与非目标

目标：

- scene compiler 不直接理解 OBJ 组织；
- 每个唯一 canonical asset 独立 realization 一次；
- 重复 scene instances 共享 `asset.xml` 和 mesh files；
- 支持一个 visual mesh 和多个 collision parts；
- asset 固有物理属性在 `asset.xml` 中完整 realization；
- 输出 package 自包含、可移动、可缓存、可独立验证；
- 删除外部 JSON asset registry 与 `asset_root` compiler 参数。

非目标：

- 在 `vsim` 中从 USDC 导出 OBJ；
- 使用 MuJoCo native OpenUSD decoder；
- 静默转换 unsupported USD geometry/material/physics；
- 本迭代支持非单位 scene instance scale；
- 通用 OBJ UV/MTL material translation；但 procedural shell 的 composed
  `UsdUVTexture` color channel 是例外：compiler 从 binding 读取 color texture，转为
  package-local PNG，并为 floor/walls 分别生成带材质的 visual geometry。normal、
  roughness、displacement channel 仍不映射到 MuJoCo。

## 4. Compiler 边界

公共入口最终收敛为：

```python
compile_csd(
    backend="mujoco",
    csd_path=Path("package/scene.usda"),
    output_root=Path("package/engine_manifests"),
    realization_config=...,
)
```

删除：

```python
asset_registry=...
asset_root=...
```

输入实际是 `scene.usd` 的完整 dependency closure，而不是只有一个文件。

## 5. Typed views

场景语义继续使用精简的 `ConcreteScenarioDefinition`，但必须保留每个 entity 的
composed prim path 与 canonical asset identity，或者提供独立映射。

资产读取层新增最小结构：

```python
@dataclass(frozen=True, slots=True)
class OpenUsdAssetResources:
    asset_id: str
    asset_digest: str
    resource_digest: str
    physics_mode: str
    visual_obj: Path
    collision_objs: tuple[Path, ...]
```

资产 realization 输出：

```python
@dataclass(frozen=True, slots=True)
class MujocoAssetRealization:
    asset_id: str
    asset_digest: str
    cache_key: str
    entry_file: Path
    generated_files: tuple[Path, ...]
```

`compiler_csd_from_openusd()` 只读取 scene semantics，不解析、复制或写资源。
OpenUSD resource reader 负责 composition result、asset resolution 和 package boundary
validation；MuJoCo asset compiler 只消费已校验 typed resources。

## 6. 两阶段 realization

```text
read/validate composed scene
          │
          ├── extract scene semantic view
          │
          └── discover unique referenced assets
                    │
                    ▼
             realize each asset
             asset.usd → asset.xml
                    │
                    ▼
             assemble scene instances
             scene.usd → scene.xml
                    │
                    ▼
             load/semantic/render diagnostics
                    │
                    ▼
             publish manifest atomically
```

Asset realization 失败时不得继续生成缺少该 entity 的 scene。

## 7. 输出目录

```text
engine_manifests/
└── mujoco/
    └── <csd_id>/
        ├── manifest.json
        ├── scene.xml
        ├── models/
        │   ├── asset-<digest-a>/
        │   │   ├── asset.xml
        │   │   ├── visual.obj
        │   │   ├── collision_000.obj
        │   │   └── collision_001.obj
        │   └── asset-<digest-b>/
        │       ├── asset.xml
        │       ├── visual.obj
        │       └── collision_000.obj
        └── diagnostics/
            ├── asset-<digest-a>-load.json
            ├── asset-<digest-b>-load.json
            ├── scene-load.json
            ├── physics.json
            ├── relationships.json
            ├── preview.png
            └── validation_record.json
```

OBJ 在 realization 中属于对应 `asset.xml` 目录，不进入全局 `assets/objects/`
registry。`asset.xml` 只使用本目录相对路径。

## 8. `asset.usd → asset.xml`

每个唯一 `(assetDigest, resourceDigest)` 生成一个完整可单独加载的 MJCF model：

```xml
<mujoco model="asset_<digest>">
  <compiler angle="radian" coordinate="local" meshdir="."/>

  <asset>
    <mesh name="visual" file="visual.obj"/>
    <mesh name="collision_000" file="collision_000.obj"/>
    <mesh name="collision_001" file="collision_001.obj"/>
  </asset>

  <worldbody>
    <body name="asset_root">
      <inertial pos="0 0 0" mass="4"
                diaginertia="0.2 0.15 0.1"/>
      <geom name="visual" type="mesh" mesh="visual"
            contype="0" conaffinity="0" density="0"/>
      <geom name="collision_000" type="mesh" mesh="collision_000"
            rgba="0 0 0 0" density="0"/>
      <geom name="collision_001" type="mesh" mesh="collision_001"
            rgba="0 0 0 0" density="0"/>
    </body>
  </worldbody>
</mujoco>
```

规则：

- asset-local transform 已在 OBJ 中，不再写 `mesh scale`；
- visual geom 不碰撞且 massless；
- collision geoms 也 massless，避免多 part 重复贡献总质量；
- mass、center of mass 与 inertia 在 `asset_root` body 层只写一次；
- friction/contact 写入对应 collision geoms；
- geom 名称稳定，attach 后由 prefix 保证实例名唯一；
- articulation asset 可以保留 asset-local body/joint hierarchy，但必须单独定义其
  物理映射与 acceptance fixture，不能把 rigid MVP 静默套用到 articulation。

`asset.xml` 必须先独立通过 `mujoco.MjModel.from_xml_path()` load check。

## 9. `scene.usd → scene.xml`

顶层 scene 注册 sub-model：

```xml
<asset>
  <model name="asset_<digest>"
         file="models/asset-<digest>/asset.xml"
         content_type="text/xml"/>
</asset>
```

每个 entity 创建 scene instance body 并 attach：

```xml
<body name="chair_01" pos="1.2 0.5 0" quat="...">
  <freejoint name="chair_01_freejoint"/>
  <attach model="asset_<digest>"
          body="asset_root"
          prefix="chair_01/"/>
</body>
```

- static entity 不添加 freejoint；
- dynamic rigid entity 添加一个 freejoint；
- prefix 从稳定 entity ID 经 MJCF name normalization 得到；
- 重复实例 attach 同一个 model asset；
- scene body 负责 world pose，asset subtree 保持资产局部空间。

## 10. 物理属性与首轮实例边界

首轮 MuJoCo realization 不引入 `MjSpec`，也不实现 attach 后的实例级 geom patch。
一个 canonical asset 的 mass、center of mass、inertia、friction 和 contact properties
必须由 composed `asset.usd` 唯一确定，并完整写入共享 `asset.xml`。

`scene.usd` 首轮只允许实例化时设置：

- entity ID/name；
- position 与 orientation；
- static/dynamic；
- task role 与 relationships。

如果 scene 对某个实例写入不同于资产值的 mass、inertia、friction 或 contact 强
opinion，MuJoCo compiler 必须返回明确 blocker。以后确有需求时，应在新迭代中选择
asset realization variant、XML 展开或程序化 model editing；本迭代不预设方案。

不得把 object 总质量重复写到每个 collision part。`asset_root` 写一次显式 inertial，
全部 visual/collision geoms 保持 massless。没有足够信息可靠产生 inertia 时必须
blocker，不能临时设计与 CSD 不一致的质量分配算法。

## 11. Cache keys

### 11.1 Asset cache key

至少包含：

```text
assetDigest
resourceDigest
target backend
asset realization config
vsim asset realization version
MuJoCo version
```

### 11.2 Scene cache key

至少包含：

```text
composed CSD digest and dependency closure
ordered asset realization cache keys
sampled randomization values
scene realization config
vsim scene realization version
MuJoCo version
```

`asset_id` 不进入唯一性判断，除非作为上述 digest 输入的一部分。

## 12. Blockers

以下情况必须返回 typed `CsdRealizationBlocker`：

- export format 未知；
- scene/asset composition 或 project semantic validation 失败；
- asset ID/digest/resource digest 缺失或冲突；
- simulation support asset path 无法解析、越界或 checksum 不符；
- rigid asset 缺 visual 或 collision OBJ；
- OBJ 格式不受 MuJoCo 支持；
- rigid scene instance scale 非单位；
- asset physics 无法映射到 MJCF；
- attach 后 name/prefix 冲突；
- scene instance 覆盖 asset 固有物理属性；
- asset.xml 或 scene.xml load check 失败；
- expected scene entity 在 realization 中缺失。

不得发布部分 manifest。

## 13. Manifest

最终 manifest 继续使用统一 backend slot，并记录：

- `entry_file = "scene.xml"`；
- scene cache key；
- 每个 asset realization ID/cache key/entry file；
- 全部 package-local dependency files；
- diagnostics 与 preview；
- validation status。

MuJoCo runtime 只消费已发布 manifest 和最终 `scene.xml`，不在启动时重新读取 CSD
或生成 asset.xml。

## 14. 从当前实现迁移

可以保留：

- `read_openusd_csd()`、variant selection 和 CSD validation；
- `ConcreteScenarioDefinition` 中已有 composed physics semantics；
- realization manifest、diagnostics 与 parity checks；
- MuJoCo backend manifest loading；
- package-local dependency discipline。

需要替换：

- `asset_registry` 与 `asset_root` public compiler parameters；
- `BackendResourceAdapter.mesh_path/collision_mesh_path` 单值模型；
- `_copy_resource_files()` 的全局 asset root 复制；
- `_write_mjcf()` 中直接注册所有 OBJ 的逻辑。

需要新增：

- OpenUSD asset dependency/resource reader；
- typed asset realization records；
- per-asset cache 与 `asset.xml` writer；
- 基于现有 XML writer 的 model registration 与 attach scene assembly；
- asset-level load diagnostics。

迁移不保留旧 JSON registry 兼容路径；项目约定允许接口断开式修改。

## 15. 测试与验收

最小测试矩阵：

1. 单 rigid asset、单 visual、单 collision；
2. 单 visual、多 collision parts；
3. 同一 asset 的两个 scene instances，共享一个 asset.xml；
4. 两实例共享 asset physics 与同一个 asset.xml；
5. scene instance physics override blocker；
6. 显式 mass/inertia 只在 body 层应用一次；
7. static 与 dynamic instance；
8. 缺失、越界或 checksum 错误的 OBJ path；
9. 非单位 instance scale blocker；
10. asset cache hit 与 resource digest invalidation；
11. scene cache hit 与 pose/static invalidation；
12. asset.xml 独立 load；
13. scene.xml load、step、nonblank render 与 entity parity。

完成标准：相关单元测试、全量测试、`ruff`、`mypy`、OpenUSD strict validation 和
MuJoCo load/physics/render diagnostics 均实际通过后，才能在实现记录中声明完成。
