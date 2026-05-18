#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ROBOSIM_ENV_NAME:-robosim}"
HABITAT_REPO="${HABITAT_SIM_REPO:-https://github.com/facebookresearch/habitat-sim.git}"
HABITAT_REF="${HABITAT_SIM_REF:-main}"
BUILD_DIR="${HABITAT_SIM_BUILD_DIR:-$(mktemp -d)}"
KEEP_BUILD_DIR="${KEEP_HABITAT_BUILD_DIR:-0}"

if [[ "${KEEP_BUILD_DIR}" != "1" ]]; then
  trap 'rm -rf "${BUILD_DIR}"' EXIT
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v mamba >/dev/null 2>&1; then
  PYTHON_BIN=(mamba run -n "${ENV_NAME}" python)
elif [[ -x "${HOME}/miniforge3/bin/mamba" ]]; then
  PYTHON_BIN=("${HOME}/miniforge3/bin/mamba" run -n "${ENV_NAME}" python)
else
  echo "Could not find an active conda env or mamba executable." >&2
  echo "Activate the robosim env first, or set ROBOSIM_ENV_NAME." >&2
  exit 1
fi

echo "Using Python:"
"${PYTHON_BIN[@]}" --version

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ ! -d habitat-sim ]]; then
  git clone --recursive --branch "${HABITAT_REF}" "${HABITAT_REPO}" habitat-sim
fi

cd habitat-sim
git submodule update --init --recursive

"${PYTHON_BIN[@]}" -m pip install "setuptools>=71,<81"
"${PYTHON_BIN[@]}" -m pip install "scikit-build-core>=0.10" "pybind11>=2.10"
"${PYTHON_BIN[@]}" -m pip install -r requirements.txt

HABITAT_BUILD_GUI_VIEWERS="${HABITAT_BUILD_GUI_VIEWERS:-ON}" \
HABITAT_WITH_BULLET="${HABITAT_WITH_BULLET:-ON}" \
"${PYTHON_BIN[@]}" -m pip install . --no-build-isolation

"${PYTHON_BIN[@]}" -c "import habitat_sim; print('habitat-sim import OK')"
