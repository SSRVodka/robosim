# MuJoCo 3.9.0 native OpenUSD feasibility

Date: 2026-07-17

## Decision

`production_mujoco_path = openusd_to_mjcf`

Building and selectively packaging MuJoCo's official USD decoder is technically
feasible, but the decoder does not preserve enough of the canonical CSD for the
accepted runtime contract. In particular, an authored `UsdGeomCamera` and
`UsdLux` light were absent from the compiled `mjModel` (`ncam = 0`,
`nlight = 0`), and the decoder contains no translation for CSD sensor objects.
The production path therefore keeps the official MuJoCo Python package and
adapts the existing compiler to read a composed OpenUSD stage and emit
package-local MJCF. The native decoder is not a second supported loader.

## Reviewed official references

- MuJoCo OpenUSD import:
  <https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html>
- Building MuJoCo with OpenUSD:
  <https://mujoco.readthedocs.io/en/stable/OpenUSD/building.html>
- MuJoCo `mjcPhysics` schemas:
  <https://mujoco.readthedocs.io/en/stable/OpenUSD/mjcPhysics.html>
- MuJoCo MJCF XML reference:
  <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- OpenUSD stages and composition:
  <https://openusd.org/release/api/class_usd_stage.html>
- OpenUSD physics schemas:
  <https://openusd.org/release/api/usd_physics_page_front.html>
- OpenUSD validation tools:
  <https://openusd.org/release/toolset.html>

The implementation probe also inspected the decoder and `mjcPhysics` sources
from the exact official MuJoCo revision recorded below.

## Reproducible inputs

| Input | Value |
| --- | --- |
| MuJoCo tag | `3.9.0` |
| MuJoCo source commit | `237c17e48539b6c90bf90d3161547cbdcbfaa1e0` |
| OpenUSD | `26.05` (`Usd.GetVersion() == (0, 26, 5)`) |
| Python | `3.12.13` |
| CMake | `4.2.3` |
| Compiler | conda-forge GCC `14.3.0` |
| Platform | Arch Linux `7.1.3`, x86-64, glibc `2.43` |
| gRPC / protobuf after install | `1.78.1` / `6.33.5` |

OpenUSD was installed in the existing `robosim2` environment and pinned in
`environment.yml`; no new environment was created. The conda-forge OpenUSD
package exposes `pxrConfig.cmake` at the environment prefix itself, so the
working configuration used `pxr_DIR=$CONDA_PREFIX`.

## Build and packaging probe

The unmodified official source was configured with:

```bash
cmake -S /tmp/robosim-mujoco-usd/source \
  -B /tmp/robosim-mujoco-usd/build-bfd \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_WITH_USD=ON \
  -Dpxr_DIR="$CONDA_PREFIX" \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF \
  -DSUPPORTS_LLD=FALSE \
  -DSUPPORTS_GOLD=FALSE \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG \
    -ffile-prefix-map=/tmp/robosim-mujoco-usd/source=/usr/src/mujoco-3.9.0 \
    -ffile-prefix-map=/tmp/robosim-mujoco-usd/build-bfd=/usr/src/mujoco-build"
cmake --build /tmp/robosim-mujoco-usd/build-bfd \
  --target usd_decoder_plugin -j 8
```

The first Release build used MuJoCo's automatically selected `lld` with
conda-forge GCC slim-LTO objects. CMake reported success, but produced tiny
unmaterialized LTO shared objects with no dynamic dependencies. Disabling both
the `lld` and `gold` feature selections made GCC use its compatible linker/LTO
plugin and produced normal shared libraries. This is a build-toolchain issue,
not an OpenUSD stage failure.

The upstream top-level install rule is unsuitable for this conda-forge layout:
it derives the OpenUSD install root as `${pxr_DIR}/lib`, then copies that entire
directory. With `pxr_DIR=$CONDA_PREFIX`, the attempted temporary install copied
about 5.8 GiB and also expected the unrelated `usdMjcf` target. The probe
therefore staged only the decoder, schema library, and schema resources; no
upstream source was edited.

Staged package inventory:

| Relative path | Size | SHA-256 |
| --- | ---: | --- |
| `lib/libusd_decoder_plugin.so` | 321880 | `db0d2424fdc9c8b2829d20473f03c603e69a77555de8ef2f95ee972de1388a4e` |
| `lib/libmjcPhysics.so` | 447288 | `5599dd2bf058c026a566703d3fc55434b5b33dcee9bbb56c0a75e3be31eb6c18` |
| `lib/mujoco-usd-resources/mjcPhysics/generatedSchema.usda` | 47033 | `04c6538ce7daba6e7b4f52cf8a580d35582bec1bc37a6d4e95f3b1cb7b0a6262` |
| `lib/mujoco-usd-resources/mjcPhysics/plugInfo.json` | 4199 | `dc1d574bd57e8076e29a527aa7924e55654003aee6e2b7c30c74199e8d09f501` |

Both shared libraries were staged with `RUNPATH=$ORIGIN`. An explicit
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` represented the package's declared runtime
dependencies while probing it outside the environment prefix. `ldd` resolved
MuJoCo 3.9.0, OpenUSD 26.05, TBB, and system libraries, and a string scan found
no `/tmp/robosim-mujoco-usd/source` or build-tree paths after applying the
compiler prefix maps.

## Stage and runtime evidence

The probe CSD is a composed stage with separate root, scenario, portable
physics, and MuJoCo variant layers. It authors meter/Z-up stage metrics,
standard rigid-body, collision, mass/inertia and revolute-joint schemas, colored
geometry, a camera, a light, gravity, and `MjcSceneAPI` on the selected MuJoCo
variant.

OpenUSD 26.05 validation passed the default stage and every backend variant:

```text
Validation Result with no explicit variants set: Success
physicsBackend:gazebo: Success
physicsBackend:mujoco: Success
physicsBackend:pybullet: Success
physicsBackend:<empty>: Success
```

In a fresh Python process, explicit
`mujoco.mj_loadPluginLibrary(libusd_decoder_plugin.so)` followed by
`mujoco.MjSpec.from_file(csd.usda)` successfully composed and compiled the
stage. The result preserved two geoms, one hinge, gravity, body mass/inertia,
renderable geometry, and finite state through 100 steps. The observed summary
was:

```json
{
  "body_mass": 1.0,
  "finite_state": true,
  "gravity": [0.0, 0.0, -9.8100004196167],
  "joint_type": 3,
  "nbody": 2,
  "ncam": 0,
  "ngeom": 2,
  "njnt": 1,
  "nlight": 0,
  "pixel_variance": 87.45605953534444
}
```

The nonblank image used a runtime free camera because the authored camera was
not present in the model. Source inspection found support for standard physics
bodies, collision, mass, joints, geometry/materials and several `mjcPhysics`
extensions (including actuators, sites, tendons, and keyframes), but no
`UsdGeomCamera` or `UsdLux` translation. Sensor support is limited to the global
MuJoCo sensor-computation flag; there is no decoder path that creates the
required sensor objects.

## Gate result

| Criterion | Result |
| --- | --- |
| Official source build | Pass after compatible linker selection |
| Selective relocatable package | Pass |
| Fresh-process plugin discovery | Pass |
| Sublayer and variant composition | Pass |
| Strict OpenUSD validation | Pass |
| Rigid body, mass/inertia, collision, joint, gravity | Pass |
| Nonblank offscreen render | Pass with runtime free camera |
| Stable forward/step | Pass |
| Authored CSD camera preserved | **Fail** |
| Authored CSD light preserved | **Fail** |
| Required CSD sensors realizable | **Fail** |

The agreed rule selects the fallback when any semantic-preservation criterion
fails. Consequently, the native package probe test was removed after recording
this evidence; it is not a production regression test. The composed OpenUSD
fixture and its strict stage-contract test remain as inputs for the selected
OpenUSD-to-MJCF implementation.

