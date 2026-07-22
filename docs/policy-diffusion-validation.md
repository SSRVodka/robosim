# Policy Runtime 变更记录：Diffusion Policy 通用通路验证

本文档记录一次独立迭代的新增内容。

## 结论

LeRobot Diffusion Policy 可以在**不修改任何 runner / adapter 代码**的前提下，
通过现有 `LerobotPolicyRunner` 通用通路（factory 加载 checkpoint +
`select_action()` + policy 内部 observation queue）在 MuJoCo 与 PyBullet 后端上
完成推理控制回路冒烟。这验证了 DESIGN.md 中"运行时适配层对其他 LeRobot IL
policy 保持通用"的设计目标对第二类 policy 成立。

## 变更内容

- `tests/test_lerobot_policy.py`：
  - checkpoint fixture 工厂 `_create_policy_checkpoint` 增加
    `policy_type: "act" | "diffusion"` 参数；
  - dataset stats fixture 统一补充 `min`/`max`（见下方差异说明）；
  - spy backend 控制回路测试与 Franka MuJoCo/PyBullet 单步冒烟测试按
    `policy_type` 参数化，新增 4 个 diffusion 用例。

无 `robosim/` 代码变更。

## 关键差异记录

- ACT 默认 MEAN_STD 归一化，stats 只需 `mean`/`std`；
- Diffusion 默认 MIN_MAX 归一化，stats 必须含 `min`/`max`，否则
  preprocessor 抛 `ValueError: MIN_MAX normalization mode requires min and max stats`；
- Diffusion 需要 `n_obs_steps` 帧观测历史，由 LeRobot policy 内部 queue 自行
  维护（首步自动复制填充），runner 每步只需提供当前观测，与 ACT 无差异。

## 后续 policy 适配约定

pi0 / pi05 / SmolVLA 等优先沿用同一 factory 通路，仅当其 observation/action
语义无法用现有 adapter 表达时才扩展 adapter。

当前阻塞：robosim 环境未安装 `transformers`，SmolVLA / pi0 的实跑验证受阻，
属于依赖缺口而非设计缺口（`get_policy_class("smolvla")` 直接
`ModuleNotFoundError: transformers`）。

## 验证证据

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_lerobot_policy.py -q
11 passed in 41.24s
```

（robosim 环境；ROS 的 launch_testing pytest 插件与当前 pytest 不兼容，需
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 规避，与本变更无关。）
