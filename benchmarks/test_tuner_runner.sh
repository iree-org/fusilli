#!/bin/bash
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
set -x

# Arguments from CMake
TUNER_SCRIPT="$1"
BENCHMARK_DRIVER="$2"
TMP_FILES=()

cleanup() {
  rm -f "${TMP_FILES[@]}"
}
trap cleanup EXIT

# This test is registered only under FUSILLI_SYSTEMS_AMDGPU; libtuner must be
# importable. If it isn't, the environment is misconfigured (e.g., test.sh's
# pip install failed) and we fail loudly rather than silently degrading.
# Mirror the import path run_tuner.py uses, since the top-level package can
# load successfully while these submodules raise on version mismatch with
# iree-compiler.
if ! python3 -c "from amdsharktuner import common, libtuner" >/dev/null 2>&1; then
  echo "ERROR: amdsharktuner is not importable; cannot run tuner integration tests."
  echo "  This test is gated on FUSILLI_SYSTEMS_AMDGPU=ON, which implies a"
  echo "  fully configured tuner environment. Check that build_tools/scripts/test.sh"
  echo "  ran the amdsharktuner install successfully."
  python3 -c "from amdsharktuner import common, libtuner" || true
  exit 1
fi

# Test 1: Verify --help works and reports the expected libtuner option groups.
HELP_OUTPUT="$(mktemp)"
TMP_FILES+=("${HELP_OUTPUT}")
python3 "${TUNER_SCRIPT}" --help > "${HELP_OUTPUT}" 2>&1
grep -q "Fusilli Tuner Options" "${HELP_OUTPUT}"
grep -q "General Options" "${HELP_OUTPUT}"
grep -q "Candidate Generation Options" "${HELP_OUTPUT}"
echo "PASSED: run_tuner.py --help"

# Cache extraction unit tests (pure-Python wrapper logic).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(cd "${SCRIPT_DIR}" && python3 -m unittest test_tuner_cache -v)
echo "PASSED: cache extraction unit tests"

# Test 2: Verify error on missing args
MISSING_ARGS_OUTPUT="$(mktemp)"
TMP_FILES+=("${MISSING_ARGS_OUTPUT}")
if python3 "${TUNER_SCRIPT}" --devices hip://0 \
  --fusilli-driver "${BENCHMARK_DRIVER}" >"${MISSING_ARGS_OUTPUT}" 2>&1; then
  echo "ERROR: Expected failure when no --fusilli-args or --commands-file given"
  exit 1
fi
grep -q "Must specify either --commands-file or --fusilli-args" "${MISSING_ARGS_OUTPUT}"
echo "PASSED: run_tuner.py rejects missing args"

# Test 3: Verify error on both args
CONFLICTING_ARGS_OUTPUT="$(mktemp)"
TMP_FILES+=("${CONFLICTING_ARGS_OUTPUT}")
if python3 "${TUNER_SCRIPT}" \
  --devices hip://0 \
  --fusilli-driver "${BENCHMARK_DRIVER}" \
  --fusilli-args "matmul -M 16 -N 16 -K 16 --a_type f32 --b_type f32 --out_type f32" \
  --commands-file /dev/null >"${CONFLICTING_ARGS_OUTPUT}" 2>&1; then
  echo "ERROR: Expected failure when both --fusilli-args and --commands-file given"
  exit 1
fi
grep -q "Cannot specify both --commands-file and --fusilli-args" "${CONFLICTING_ARGS_OUTPUT}"
echo "PASSED: run_tuner.py rejects conflicting args"

echo "ALL TESTS PASSED"
