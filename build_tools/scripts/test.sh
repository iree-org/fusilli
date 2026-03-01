#!/usr/bin/env bash

# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<EOF
Usage: test.sh [options]

Options:
  --build-dir DIR            Build directory (default: build/)
  --timeout SECS             Test timeout in seconds (default: 120)
  --parallel N               Number of parallel tests (default: \$(nproc))
  --backend capi|cli         Compile backend (default: capi)
  --extra-verbose            Print extra test output (default: off)
  --skip-cache-validation    Skip running test_cache_empty.sh after tests
EOF
  exit 1
}

BUILD_DIR="build"
TIMEOUT=120
PARALLEL="$(nproc)"
BACKEND="capi"
EXTRA_VERBOSE=false
SKIP_CACHE_VALIDATION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --extra-verbose)
      EXTRA_VERBOSE=true
      shift
      ;;
    --skip-cache-validation)
      SKIP_CACHE_VALIDATION=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

if [[ "${BACKEND}" == "cli" ]]; then
  export FUSILLI_COMPILE_BACKEND_USE_CLI=1
  echo "=== Fusilli test: backend=cli ==="
else
  echo "=== Fusilli test: backend=capi ==="
fi

CTEST_ARGS=(
  --test-dir "${BUILD_DIR}"
  --output-on-failure
  --timeout "${TIMEOUT}"
  -j "${PARALLEL}"
)

if [[ "${EXTRA_VERBOSE}" == "true" ]]; then
  CTEST_ARGS+=(--extra-verbose)
fi

ctest "${CTEST_ARGS[@]}"

if [[ "${SKIP_CACHE_VALIDATION}" == "false" ]]; then
  "${REPO_ROOT}/tests/test_cache_empty.sh"
fi
