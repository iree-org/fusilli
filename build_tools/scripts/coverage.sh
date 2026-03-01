#!/usr/bin/env bash

# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

usage() {
  cat <<EOF
Usage: coverage.sh [options]

Options:
  --build-dir DIR     Build directory (default: build/)
  --timeout SECS      Test timeout in seconds (default: 120)
  --output-dir DIR    Coverage report output directory (default: coverage_report/)
EOF
  exit 1
}

BUILD_DIR="build"
TIMEOUT=120
OUTPUT_DIR="coverage_report"

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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

echo "=== Fusilli coverage: build-dir=${BUILD_DIR} output-dir=${OUTPUT_DIR} ==="

ctest --test-dir "${BUILD_DIR}" -T test -T coverage --timeout "${TIMEOUT}"

lcov --capture \
     --directory "${BUILD_DIR}" \
     --output-file "${BUILD_DIR}/coverage.info"

lcov --remove "${BUILD_DIR}/coverage.info" \
     '/usr/*' '*/iree/*' \
     --output-file "${BUILD_DIR}/coverage.info"

genhtml "${BUILD_DIR}/coverage.info" \
        --output-directory "${OUTPUT_DIR}"
