# OpenUSD CSD Package 与资产交接规约

## 1. 范围

本文定义 AssetServer 向 `vsim` 交付 OpenUSD CSD snapshot 的最小契约。当前格式为：

```text
scene-export/v9-vsim-articulated-resources
```

层级固定为：

```text
scene.usda  --USD reference-->  asset.usda  --custom asset fields-->  OBJ
```

USD package 是语义来源；OBJ 是 export worker 从 composed USD 生成的仿真支持文件。
`vsim` 只接收 `scene.usda` 入口及其包内 dependency closure，不使用额外资产 registry。

## 2. Package 目录

```text
package/
├── scene.usda
├── manifest.json
├── checksums.sha256
├── workspace.json
├── previews/
├── assets/
│   ├── asset-<rigid-digest>/
│   │   ├── asset.usda
│   │   ├── geometry.usdc
│   │   ├── collision.usdc
│   │   └── support/obj/
│   │       ├── visual.obj
│   │       └── collision_NNN.obj
│   └── asset-<articulated-digest>/
│       ├── asset.usda
│       ├── geometry.usdc
│       ├── collision.usdc
│       └── support/obj/links/<link-key>/
│           ├── visual.obj
│           └── collision_NNN.obj
└── shells/
    └── <shell-key>/
        ├── asset.usda
        ├── geometry.usdc
        └── support/obj/
            ├── visual.obj
            └── collision_NNN.obj
```

资产目录还可包含 manifest、metadata、validation、texture 和 preview。所有
`checksums.sha256` 条目必须在发布前通过校验。路径必须为包内 POSIX 相对路径，
不得包含 `..`、绝对路径或外部 cache 路径。

同一资产多次实例化时只复制一份资产目录；实例 transform 只存在于 `scene.usda`。

## 3. Scene composition

```usda
def Xform "cabinet_01" (
    prepend references =
        @assets/asset-<digest>/asset.usda@
)
{
    double3 xformOp:translate = (1.2, 0.5, 0)
    quatf xformOp:orient = (1, 0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}
```

场景负责实例 ID、pose、环境和初始状态。资产固有的 geometry、mass、inertia、
contact material、link 和 joint 必须由 `asset.usda` 定义。首轮 MuJoCo compiler
不接受 scene 对资产固有物理属性的覆盖；场景中的初始 joint target 属于实例状态，
不改变资产定义。

### 3.1 Robot descriptor

一个 v9 scene 至多包含一个直接位于 `/World/Robot` 的固定基座 robot descriptor。
它是场景声明，不是 `assets/` 中的 USD reference：MuJoCo compiler 根据
`robosim:robot:id` 选择 compiler-owned control template，复制其 MJCF、SRDF 与
mesh dependency closure，并把本 prim 的 pose patch 到复制后模板的 root body。

```usda
def Xform "Robot"
{
    custom string robosim:robot:id = "franka_panda"
    custom string robosim:robot:instanceId = "robot"
    quatd xformOp:orient = (1, 0, 0, 0)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}
```

`robot:id` 是 target backend 支持的 template identity，不是普通 asset ID；当前
MuJoCo target 支持 `franka_panda`。`robot:instanceId` 是稳定场景实例名。robot
不得有非单位 scale。模板 closure hash 由 compiler 加入 realization cache key；输出
不得在运行时依赖 template source。模板中的未声明 camera/light 不得泄漏到 scene。

### 3.2 Cameras and lights

scene 可以在 `/World/Cameras` 下直接声明 `Camera`，在 `/World/Lights` 下直接声明
`DistantLight`。prim name 是 runtime sensor/light name；位姿使用标准 translate 与
orientation（或 `rotateXYZ`）。

```usda
def Scope "Cameras"
{
    def Camera "agent_view"
    {
        float focalLength = 24
        float verticalAperture = 20.25
        quatd xformOp:orient = (0.82, 0.425, 0.176, 0.34)
        double3 xformOp:translate = (1.4, -1.4, 1.4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
}

def Scope "Lights"
{
    def DistantLight "key_light"
    {
        float intensity = 1
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = (0, 0, 3)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
}
```

MuJoCo realization 将 camera pose 转为 MJCF `pos`/`xyaxes`，并由
`focalLength`/`verticalAperture` 计算 `fovy`。DistantLight 生成 directional MJCF
light，orientation 决定照射方向，`intensity` 映射为 RGB diffuse intensity。没有
authored camera 时，compiler 才生成仅用于 preview 的 `world_camera` fallback。
已 authored camera 在 runtime 以同名 `CAMERA` sensor 出现在 `ListSensors`，并可由
`GetSensors([name])` 返回 `rgb8` 图像；当前 CameraImage contract 不传回 intrinsics。

## 4. 公共资产字段

每个资产 default prim 必须声明：

```usda
custom string assetserver:asset:id = "asset://sha256/<digest>"
custom token assetserver:simulationSupport:physicsMode = "rigid"
custom string assetserver:simulationSupport:resourceDigest = "sha256:<digest>"
```

- `asset:id` 是该不可变 canonical/export asset revision 的身份；普通资产使用
  `asset://sha256/...`，procedural shell 可使用 `shell://sha256/...`。
- ID 中 digest 参与内容寻址和去重；compiler 不从目录名推断 ID。
- `resourceDigest` 校验 OBJ 支持资源并参与 realization cache 失效。
- `sourceDigest`、`sourceAssetDigest` 和 `assetInfo.identifier` 仅为 provenance，
  不作为 compiler 资产身份。
- `physicsMode` 当前允许 `rigid`、`articulated`、`static`。

所有 dynamic body 必须有确定的 mass、center of mass、diagonal inertia 和 principal
axes。所有 collision 必须绑定资产级默认 `PhysicsMaterialAPI`；link 可给出更强的
资产内部 opinion。不得要求 scene 补齐这些字段。

## 5. Rigid 格式

Rigid 资产在 default prim 上声明聚合 visual、多个 collision part 和完整刚体属性：

```usda
def Xform "Asset" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    kind = "component"
)
{
    custom string assetserver:asset:id = "asset://sha256/<digest>"
    custom token assetserver:simulationSupport:physicsMode = "rigid"
    custom asset assetserver:simulationSupport:visualObj =
        @support/obj/visual.obj@
    custom asset[] assetserver:simulationSupport:collisionObjs = [
        @support/obj/collision_000.obj@,
        @support/obj/collision_001.obj@
    ]
    custom string assetserver:simulationSupport:resourceDigest = "sha256:<digest>"

    float physics:mass = 4
    point3f physics:centerOfMass = (0, 0, 0.4)
    float3 physics:diagonalInertia = (0.2, 0.15, 0.1)
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physics:rigidBodyEnabled = true

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    ) {
        float physics:staticFriction = 0.5
        float physics:dynamicFriction = 0.5
        float physics:restitution = 0
    }
}
```

## 6. Articulated 格式

Articulated 资产的 root 声明 articulation；每个 link 独立持有 OBJ 和惯性，joint
使用标准 USD Physics relationship 连接 link：

```text
asset-<digest>/
├── asset.usda
└── support/obj/links/
    ├── Links_Base/
    │   ├── visual.obj
    │   └── collision_000.obj
    └── Links_Door/
        ├── visual.obj
        ├── collision_000.obj
        └── collision_001.obj
```

```usda
def Xform "Asset" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    kind = "component"
)
{
    custom string assetserver:asset:id = "asset://sha256/<digest>"
    custom token assetserver:simulationSupport:physicsMode = "articulated"
    custom string assetserver:simulationSupport:resourceDigest = "sha256:<digest>"

    def Scope "Links"
    {
        def Xform "Base" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        ) {
            rel material:binding:physics = </Asset/PhysicsMaterial>
            custom asset assetserver:simulationSupport:visualObj =
                @support/obj/links/Links_Base/visual.obj@
            custom asset[] assetserver:simulationSupport:collisionObjs = [
                @support/obj/links/Links_Base/collision_000.obj@
            ]
            float physics:mass = 20
            point3f physics:centerOfMass = (0, 0, 0.5)
            float3 physics:diagonalInertia = (1, 2, 2)
            quatf physics:principalAxes = (1, 0, 0, 0)
        }

        def Xform "Door" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        ) {
            rel material:binding:physics = </Asset/PhysicsMaterial>
            custom asset assetserver:simulationSupport:visualObj =
                @support/obj/links/Links_Door/visual.obj@
            custom asset[] assetserver:simulationSupport:collisionObjs = [
                @support/obj/links/Links_Door/collision_000.obj@,
                @support/obj/links/Links_Door/collision_001.obj@
            ]
            float physics:mass = 2.5
            point3f physics:centerOfMass = (0, 0, 0)
            float3 physics:diagonalInertia = (0.02, 0.08, 0.1)
            quatf physics:principalAxes = (1, 0, 0, 0)
        }
    }

    def Scope "Joints"
    {
        def PhysicsRevoluteJoint "DoorJoint"
        {
            rel physics:body0 = </Asset/Links/Base>
            rel physics:body1 = </Asset/Links/Door>
            uniform token physics:axis = "Z"
            point3f physics:localPos0 = (0.3, 0, 0.8)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
    }

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    ) {
        float physics:staticFriction = 0.5
        float physics:dynamicFriction = 0.5
        float physics:restitution = 0
    }
}
```

每个 link 的 `visualObj` 恰好一个，`collisionObjs` 至少一个。数组而非文件编号是
权威对应关系。凸分解 part 数量应有后端上限；compiler 不合并或重新分解 collision。

## 7. Static 格式

Static 用于 room shell 等固定环境。它有几何和 contact material，但不是动态刚体：

```text
shells/<shell-key>/
├── asset.usda
├── geometry.usdc
└── support/obj/
    ├── visual.obj
    ├── collision_000.obj
    └── collision_001.obj
```

```usda
def Xform "Asset" (kind = "component")
{
    custom string assetserver:asset:id = "shell://sha256/<digest>"
    custom token assetserver:simulationSupport:physicsMode = "static"
    custom asset assetserver:simulationSupport:visualObj =
        @support/obj/visual.obj@
    custom asset[] assetserver:simulationSupport:collisionObjs = [
        @support/obj/collision_000.obj@,
        @support/obj/collision_001.obj@
    ]
    custom string assetserver:simulationSupport:resourceDigest = "sha256:<digest>"
    rel material:binding:physics = </Asset/PhysicsMaterial>

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    ) {
        float physics:staticFriction = 0.5
        float physics:dynamicFriction = 0.5
        float physics:restitution = 0
    }
}
```

Static root 不得应用 `PhysicsRigidBodyAPI`，`vsim` 不为其创建 free joint。即使 USD
中保留质量 metadata，compiler 也不将其 realization 为动态惯性。

## 8. OBJ 与 transform

- `visualObj` 是资产或 link 的聚合 visual mesh。
- `collisionObjs` 按稳定顺序列出独立 collision parts；不得扫描目录猜测对应关系。
- OBJ 至少承诺 `v` 和 `f`，不承诺 normals、UV、MTL、texture 或 deformation。
- 每个 OBJ 的 composed transform 必须烘焙到所属 asset root 或 articulated link 的
  局部坐标系，并规范化到 `metersPerUnit = 1`、`upAxis = "Z"`。
- OBJ 不包含 scene instance transform；首轮 compiler 要求实例 scale 为 `(1,1,1)`。
- `vsim` 不再次应用 mesh scale，也不以 visual mesh 替代缺失 collision。

## 9. Validation

遇到以下情况必须 validation failure，不得静默降级：

- ID、physics mode、resource digest 或必需 OBJ 缺失；
- digest/checksum 不匹配，或引用越出 package；
- dynamic body 的质量或惯性字段不完整、非法；
- collision 未绑定资产内定义的物理材质；
- joint 引用不存在的 link，或 joint 类型/字段不能无损映射；
- scene 覆盖资产固有物理属性；
- 非单位 instance scale 或不支持的 geometry/physics feature。

MuJoCo realization 的输出和缓存规则见
[`mujoco-openusd-asset-compiler.md`](./mujoco-openusd-asset-compiler.md)。
