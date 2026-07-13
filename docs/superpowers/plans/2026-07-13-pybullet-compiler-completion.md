# PyBullet Compiler Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring PyBullet CSD compiler/runtime completeness up to the current MuJoCo compiler level for robot-bearing tabletop CSDs.

**Architecture:** Keep the generated PyBullet scene as a package made of `manifest.json`, `scene.py`, `scene_meta.json`, local `assets/`, and `diagnostics/`. Add explicit PyBullet robot realization for Franka Panda by copying the URDF dependency closure into the package, loading it from `scene.py`, and naming it in metadata so `PyBulletBackend` exposes robot state/spec/control when loading a compiled manifest.

**Tech Stack:** Python 3, PyBullet, MuJoCo Python bindings for visual comparison, pytest, ruff, mypy.

## Global Constraints

- Documentation for an iteration must be updated before implementation begins or after it concludes.
- Do not modify MuJoCo backend behavior to satisfy PyBullet acceptance.
- Do not silently downgrade CSD semantics. Missing PyBullet robot support must return a typed blocker.
- Generated PyBullet packages must not depend on source `drivers_sim` paths or external PyBullet data search paths for realized robot assets.
- Keep implementation narrow; split compiler code only if the PyBullet-specific changes make `robosim/core/csd_compiler.py` materially harder to review.

---

### Task 1: Robot-Bearing CSD Red Tests

**Files:**
- Modify: `tests/test_pybullet_csd_compiler.py`
- Modify: `tests/test_pybullet_backend.py`
- Modify: `tests/fixtures/csd/asset_registry_pybullet.json`

**Interfaces:**
- Consumes: `compile_csd_to_pybullet(...)`, `PyBulletBackend.from_csd_realization_manifest(...)`.
- Produces: failing tests that require `robot_franka_panda` to appear in PyBullet metadata, generated files, and runtime robot spec.

- [ ] Add a compiler test using `franka_tabletop_single_object.json` that asserts `scene_meta.json` includes `robot_name == "panda"`, package-local robot files under `assets/robots/`, and generated scene body names include `panda`.
- [ ] Add a runtime test that loads the same manifest and asserts `get_robot_spec().robot_name == "panda"`, `panda_arm` exists, and `mug` still exists in `body_names`.
- [ ] Run the two tests and verify they fail because the compiler omits the robot.

### Task 2: Minimal PyBullet Robot Realization

**Files:**
- Modify: `robosim/core/csd_compiler.py`
- Modify: `tests/fixtures/csd/asset_registry_pybullet.json`

**Interfaces:**
- Consumes: `scenario.robot.asset_id` from CSD.
- Produces: generated `scene_meta.json` robot fields and package-local robot URDF/assets for `robot_franka_panda`.

- [ ] Add PyBullet robot compatibility handling for `robot_franka_panda`/`franka_panda`; unsupported robot assets return `CsdRealizationBlocker`.
- [ ] Copy the PyBullet Panda URDF dependency closure from installed `pybullet_data` into `assets/robots/franka_panda/`.
- [ ] Emit robot metadata with `robot_name`, `urdf_path`, `position`, `orientation_xyzw`, and `fixed_base`.
- [ ] Update generated `scene.py` to load the robot from the package-local URDF before objects.
- [ ] Run the red tests and verify they pass.

### Task 3: Visual Similarity Verification

**Files:**
- Modify: `tests/test_pybullet_csd_compiler.py`
- Create or modify: a small diagnostics helper only if existing helper code is insufficient.

**Interfaces:**
- Consumes: MuJoCo and PyBullet compiler outputs for `franka_tabletop_single_object.json`.
- Produces: deterministic preview artifacts and a documented visual check result.

- [ ] Generate MuJoCo and PyBullet realization packages for the same CSD fixture.
- [ ] Produce previews from each package without changing backend behavior.
- [ ] Add a lightweight automated sanity check that both previews are nonblank and have comparable frame occupancy.
- [ ] Record the visual verification command/result in `TODO.md` after the iteration concludes.

### Task 4: Final Verification

**Files:**
- Modify only if verification exposes narrow PyBullet bugs.

**Interfaces:**
- Consumes: all PyBullet compiler/runtime tests.
- Produces: verified acceptance evidence.

- [ ] Run focused PyBullet tests.
- [ ] Run `ruff check .`.
- [ ] Run `mypy .`.
- [ ] Run README smoke commands against `python -m robosim.server --backend pybullet --headless`.
- [ ] Commit the completed iteration with a scoped message.
