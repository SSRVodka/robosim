# PyBullet Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full PyBullet compiler/runtime support that satisfies the README examples through the existing gRPC, servo, recorder/replay, and CSD realization stack.

**Architecture:** PyBullet mirrors the MuJoCo architecture and validation discipline, but uses a PyBullet-specific realization package. URDF files represent physical bodies, `scene.py` deterministically assembles the PyBullet world, and `scene_meta.json` records sensors, cameras, body mappings, and validation expectations.

**Tech Stack:** Python 3.12, PyBullet, gRPC stubs, LeRobot 0.5.1, pytest, ruff, mypy.

## Global Constraints

- Use the existing `robosim2` miniforge environment; do not create or modify environments.
- Do not change fixed gRPC versions: `grpcio==1.78.1`, `protobuf==6.33.5`.
- Generated backend packages must be self-contained under `engine_manifests/pybullet/<csd_id>/`.
- Generated Python scene loaders are compiler-owned deterministic artifacts, not an open extension surface.
- CSD remains the semantic source; URDF, scripts, and metadata are reproducible cache artifacts.
- Unsupported or lossy backend semantics must return typed blockers instead of silent degradation.

---

### Task 1: Documentation And Public Surface

**Files:**
- Modify: `DESIGN.md`
- Modify: `README.md`
- Modify: `TODO.md`
- Modify: `robosim/backends/__init__.py`
- Modify: `robosim/server.py`

**Interfaces:**
- Produces: user-facing `--backend pybullet` entry point and documented package layout.

- [x] **Step 1: Record PyBullet realization design**

Add the URDF + `scene.py` + `scene_meta.json` contract to `DESIGN.md`, and update README/TODO to list PyBullet as a backend target.

- [x] **Step 2: Add server/backend exports**

Expose `PyBulletBackend` from `robosim.backends`, add `pybullet` to CLI choices, and instantiate it from either default scene data or `--csd-manifest`.

- [x] **Step 3: Verify docs and CLI import path**

Run: `python -m py_compile robosim/server.py robosim/backends/__init__.py`

Expected: command exits with status 0.

### Task 2: PyBullet CSD Compiler

**Files:**
- Modify: `robosim/core/csd_compiler.py`
- Modify: `robosim/core/__init__.py`
- Test: `tests/test_pybullet_csd_compiler.py`
- Test fixture: `tests/fixtures/csd/asset_registry_pybullet.json`

**Interfaces:**
- Produces: `compile_csd_to_pybullet(...) -> CsdCompilationResult`
- Produces: `compile_csd(..., backend="pybullet", ...)`
- Produces: package files `manifest.json`, `scene.py`, `scene_meta.json`, `assets/`, `diagnostics/`

- [x] **Step 1: Write failing compiler package test**

Assert that compiling `object_only_static_and_dynamic.json` with a PyBullet asset registry writes the expected package, copies mesh assets, writes metadata cameras/object mappings, and returns a PyBullet manifest.

- [x] **Step 2: Run compiler test red**

Run: `pytest tests/test_pybullet_csd_compiler.py::test_compile_csd_to_pybullet_writes_self_contained_package -q`

Expected: FAIL because `compile_csd_to_pybullet` is not defined/exported.

- [x] **Step 3: Implement minimal compiler**

Add a PyBullet branch that validates backend adapters, writes package-local OBJ resources, emits generated object URDF files, writes deterministic `scene.py`, writes `scene_meta.json`, and persists `manifest.json`.

- [x] **Step 4: Add validation smoke**

Load the generated package in PyBullet DIRECT, run a short finite-state physics check, render a nonblank preview from the first CSD camera, and write diagnostics/validation record.

- [x] **Step 5: Run compiler tests green**

Run: `pytest tests/test_pybullet_csd_compiler.py -q`

Expected: PASS.

### Task 3: PyBullet Runtime Backend

**Files:**
- Create: `robosim/backends/pybullet/__init__.py`
- Create: `robosim/backends/pybullet/backend.py`
- Test: `tests/test_pybullet_backend.py`

**Interfaces:**
- Produces: `PyBulletBackend(scene_path: str | None = None, scene_meta_path: str | None = None, headless: bool = True)`
- Produces: `PyBulletBackend.from_csd_realization_manifest(...)`
- Produces: `PyBulletBackend.from_csd_realization_manifest_file(...)`

- [x] **Step 1: Write failing backend startup/spec/sensor tests**

Assert default headless backend starts, returns Franka robot spec, lists `joint_states` and `world_camera`, renders a nonblank image, and shuts down cleanly.

- [x] **Step 2: Run backend tests red**

Run: `pytest tests/test_pybullet_backend.py::test_default_pybullet_backend_starts_and_renders -q`

Expected: FAIL because `PyBulletBackend` does not exist.

- [x] **Step 3: Implement minimal runtime**

Create a PyBullet DIRECT/GUI client, load default Franka URDF from `pybullet_data`, maintain a stepping thread, build joint/group/sensor registries, and implement state/spec/sensor/reset/shutdown methods.

- [x] **Step 4: Add control and servo tests**

Assert position control moves a joint target, `get_joint_command_state()` exposes replayable positions, end-effector FK returns a pose, and joint servo stream yields state.

- [x] **Step 5: Implement control/servo**

Use `setJointMotorControlArray` for position/velocity/torque modes, `getLinkState` for FK, and `calculateJacobian` plus damped least squares for twist commands.

- [x] **Step 6: Run backend tests green**

Run: `pytest tests/test_pybullet_backend.py -q`

Expected: PASS.

### Task 4: Full Stack Wiring

**Files:**
- Modify: `robosim/server.py`
- Modify: `robosim/grpc_server/simulation.py`
- Test: `tests/test_server.py`
- Test: `tests/test_lerobot_recorder.py`

**Interfaces:**
- Consumes: `PyBulletBackend`
- Produces: gRPC server startup path and recorder/replay compatibility.

- [x] **Step 1: Write failing server and recorder smoke tests**

Assert `create_backend(backend_type="pybullet", ...)` returns `PyBulletBackend`, and LeRobot recorder can record/end/replay one short PyBullet episode.

- [x] **Step 2: Run full-stack tests red**

Run: `pytest tests/test_server.py tests/test_lerobot_recorder.py -q`

Expected: FAIL on missing PyBullet server wiring or runtime methods.

- [x] **Step 3: Implement server wiring**

Add `pybullet` CLI choice, `--csd-manifest` support for PyBullet, and route simulation methods to backend optional methods where implemented.

- [x] **Step 4: Run focused full-stack tests green**

Run: `pytest tests/test_server.py tests/test_lerobot_recorder.py tests/test_pybullet_backend.py tests/test_pybullet_csd_compiler.py -q`

Expected: PASS.

### Task 5: Final Verification

**Files:**
- No new files unless verification exposes a focused fix.

**Interfaces:**
- Produces: acceptance evidence.

- [ ] **Step 1: Run lint**

Run: `ruff check .`

Expected: PASS.

- [ ] **Step 2: Run type check**

Run: `mypy .`

Expected: PASS.

- [ ] **Step 3: Run tests**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 4: Smoke README examples**

Run PyBullet backend startup/import path and CLI tools against a local server where practical:
`python -m robosim.server --backend pybullet --port 50051 --headless`,
`python -m control_stubs.tools.servo_keyboard --list`,
`python -m control_stubs.tools.data_recorder start ...`,
`python -m control_stubs.tools.data_recorder end`,
`python -m control_stubs.tools.data_recorder replay ...`.

Expected: commands complete without backend-specific errors.
