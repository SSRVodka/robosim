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
- [ ] Delete JSON-only tests and helpers only after their semantic replacements
      pass.

## Checkpoint 3: PyBullet

- [x] Adapt PyBullet URDF/scene/meta generation to the same OpenUSD fixture.
- [x] Match the shared transform, mass/inertia, collision, gravity, camera,
      sensor, material, relationship/blocker, render, and stepping checks.

## Checkpoint 4: Gazebo

- [ ] Complete OpenUSD-to-SDFormat realization for the shared fixture.
- [ ] Validate generated SDFormat with the installed official tools and load it
      in Gazebo 11 headlessly.
- [ ] Add package-local asset, physics, camera/light/sensor, blocker, manifest,
      and runtime handoff evidence.

## Checkpoint 5: Three-backend acceptance

- [ ] Remove the legacy CSD JSON schema, persistence, fixtures, and compiler
      entry points after all semantic coverage has migrated.
- [ ] Verify strict OpenUSD validation, all project tests, ruff, mypy, generated
      protobuf consistency, and a clean repository diff.
- [ ] Commit each checkpoint in `vsim` first and then update the parent
      submodule pointer and synchronized thesis documentation.
