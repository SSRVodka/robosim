# OpenUSD CSD implementation milestones

The canonical design is fixed in `DESIGN.md`. Do not change these milestones
during an implementation checkpoint without user approval.

## Checkpoint 1: OpenUSD handoff contract

- [x] Package the project codeless CSD schemas and registration helper.
- [x] Add the composed-stage path and typed OpenUSD reader; migrate the public
      compiler input in Checkpoint 2.
- [x] Hash and validate the composed CSD, its layers, all backend variants, and
      resolved dependencies.
- [x] Make the benchmark generator persist `csd/<csd_id>/csd.usda` plus layers,
      with no equivalent CSD JSON.
- [x] Add shared composed fixtures covering environment, robot, objects,
      material/physics, relationships, sampled overrides, sensors, evaluators,
      and all three backend variants.

## Checkpoint 2: MuJoCo

- [x] Adapt the existing MJCF compiler and cache to the typed OpenUSD reader.
- [x] Preserve the accepted MuJoCo load, semantic, relationship, physics,
      render, validation-record, runtime-manifest, and package-local asset tests.
- [x] Delete JSON-only tests and helpers only after their semantic replacements
      pass.

## Checkpoint 3: PyBullet

- [x] Adapt PyBullet URDF/scene/meta generation to the same OpenUSD fixture.
- [x] Match the shared transform, mass/inertia, collision, gravity, camera,
      sensor, material, relationship/blocker, render, and stepping checks.

## Checkpoint 4: Gazebo

- [x] Complete OpenUSD-to-SDFormat realization for the shared fixture.
- [x] Validate generated SDFormat with the installed official tools and load it
      in Gazebo 11 headlessly.
- [x] Add package-local asset, physics, camera/light/sensor, blocker, manifest,
      and runtime handoff evidence.

## Checkpoint 5: Three-backend acceptance (reopened)

- [x] Remove the legacy CSD JSON schema, persistence, fixtures, and compiler
      entry points after all semantic coverage has migrated.
- [x] Verify strict OpenUSD validation, all project tests, ruff, mypy, generated
      protobuf consistency, and a clean repository diff.
- [x] Commit each checkpoint in `vsim` first and then update the parent
      submodule pointer and synchronized thesis documentation.
- [ ] Enforce scene-entity parity: backend compilers must not add physical or
      rendered entities that are absent from the selected OpenUSD stage.
- [ ] Apply the authored robot pose to the copied MuJoCo robot template and
      reject regressions with native-model and preview-visibility assertions.
- [ ] Make shared test asset dimensions agree with the authored OpenUSD geometry.
- [ ] Compile and load the portable four-scene acceptance matrix in MuJoCo,
      PyBullet, and Gazebo; require deterministic preview visibility from MuJoCo
      and PyBullet and official SDF/headless-load evidence from Gazebo Classic 11.
- [ ] Re-run strict OpenUSD validation, all project tests, ruff, mypy, and the
      generated-protobuf consistency check before closing this checkpoint again.
