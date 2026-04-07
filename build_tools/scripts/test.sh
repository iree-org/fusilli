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
  --repeat N                 Repeat each test N times, stop on first failure
  -R REGEX                   Only run tests matching regex
  -E REGEX                   Exclude tests matching regex
  --extra-verbose            Print extra test output (default: off)
  --validate-cache-cleanup   Run test_cache_empty.sh after tests (default: off)
EOF
  exit 1
}

BUILD_DIR="build"
TIMEOUT=120
PARALLEL="$(nproc)"
BACKEND="capi"
REPEAT=""
EXTRA_VERBOSE=false
VALIDATE_CACHE_CLEANUP=false
INCLUDE_REGEX=""
EXCLUDE_REGEX=""

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
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    -R)
      INCLUDE_REGEX="$2"
      shift 2
      ;;
    -E)
      EXCLUDE_REGEX="$2"
      shift 2
      ;;
    --extra-verbose)
      EXTRA_VERBOSE=true
      shift
      ;;
    --validate-cache-cleanup)
      VALIDATE_CACHE_CLEANUP=true
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

if [[ -n "${INCLUDE_REGEX}" ]]; then
  CTEST_ARGS+=(-R "${INCLUDE_REGEX}")
fi

if [[ -n "${EXCLUDE_REGEX}" ]]; then
  CTEST_ARGS+=(-E "${EXCLUDE_REGEX}")
fi

if [[ -n "${REPEAT}" ]]; then
  CTEST_ARGS+=(--repeat until-fail:"${REPEAT}")
fi

if [[ "${EXTRA_VERBOSE}" == "true" ]]; then
  CTEST_ARGS+=(--extra-verbose)
fi

ctest "${CTEST_ARGS[@]}"

if [[ "${VALIDATE_CACHE_CLEANUP}" == "true" ]]; then
  "${REPO_ROOT}/tests/test_cache_empty.sh"
fi
