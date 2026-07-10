#!/usr/bin/env bash
#
# bootstrap.sh — provision dependencies and build Modern-Computational-Nonlinear-Filtering (NLF).
#
# NLF requires its sibling dependency, OptimizedKernelsForRaspberryPi5_NvidiaCUDA
# (OptMathKernels), to be present on disk. This script checks for it and, if it is
# missing, OFFERS to git-clone it (latest `main`) over HTTPS and then builds
# everything for you. It never fetches from the network without asking.
#
# Usage:
#   ./bootstrap.sh [extra cmake args...]
#   FORCE_YES=1 ./bootstrap.sh          # non-interactive: assume "yes" to clones
#
# Extra args after the script name are passed through to the CMake configure step,
# e.g.:  ./bootstrap.sh -DNLF_BUILD_PYTHON_VENV=OFF
#
set -euo pipefail

# --- Configuration ---------------------------------------------------------
OPTMATH_NAME="OptimizedKernelsForRaspberryPi5_NvidiaCUDA"
NLF_NAME="Modern-Computational-Nonlinear-Filtering"
OPTMATH_URL="https://github.com/n4hy/${OPTMATH_NAME}.git"
NLF_URL="https://github.com/n4hy/${NLF_NAME}.git"
BRANCH="main"

# --- Locate the NLF repo root ---------------------------------------------
# If run from inside a checkout, ROOT is this script's directory. If someone
# curl-pipes this script standalone (no NLF checkout), offer to clone NLF first.
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -n "$SCRIPT_SOURCE" && -f "$SCRIPT_SOURCE" ]]; then
    ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
else
    ROOT=""
fi

# Prompt helper: honours FORCE_YES=1 and non-interactive stdin (defaults to No).
confirm() {
    local prompt="$1"
    if [[ "${FORCE_YES:-0}" == "1" ]]; then
        echo "${prompt} [auto-yes]"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        echo "${prompt} [non-interactive, assuming No]"
        return 1
    fi
    local reply
    read -r -p "${prompt} [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

require_git() {
    command -v git >/dev/null 2>&1 || { echo "ERROR: git is required but not on PATH." >&2; exit 1; }
    command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required but not on PATH." >&2; exit 1; }
}

require_git

# --- Ensure the NLF repo itself exists (standalone-run case) ---------------
if [[ -z "$ROOT" || ! -f "$ROOT/CMakeLists.txt" ]]; then
    echo "This script was not run from inside a ${NLF_NAME} checkout."
    if confirm "Clone ${NLF_URL} (${BRANCH}) into ./${NLF_NAME}?"; then
        git clone --branch "$BRANCH" "$NLF_URL" "$NLF_NAME"
        ROOT="$(cd "$NLF_NAME" && pwd)"
    else
        echo "Aborting. Clone it manually with:" >&2
        echo "    git clone $NLF_URL" >&2
        exit 1
    fi
fi

# --- Ensure the OptMath sibling exists -------------------------------------
OPTMATH_DIR="$(cd "$ROOT/.." && pwd)/${OPTMATH_NAME}"

if [[ -f "$OPTMATH_DIR/CMakeLists.txt" ]]; then
    echo "Found required dependency OptMathKernels at: $OPTMATH_DIR"
else
    echo "Required dependency OptMathKernels was NOT found at:"
    echo "    $OPTMATH_DIR"
    if confirm "Clone ${OPTMATH_URL} (${BRANCH}) there now?"; then
        git clone --branch "$BRANCH" "$OPTMATH_URL" "$OPTMATH_DIR"
    else
        echo "Aborting. Provision it manually with:" >&2
        echo "    git clone $OPTMATH_URL \"$OPTMATH_DIR\"" >&2
        exit 1
    fi
fi

# --- Configure and build ---------------------------------------------------
BUILD_DIR="$ROOT/build"
echo "Configuring (Release) in $BUILD_DIR ..."
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAUTO_CLONE_DEPS=OFF \
    -DOPTMATH_DIR="$OPTMATH_DIR" \
    "$@"

echo "Building with $(nproc) jobs ..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Build complete. Binaries are under: $BUILD_DIR"
echo "Run the test suite with:  ctest --test-dir \"$BUILD_DIR\" --output-on-failure"
