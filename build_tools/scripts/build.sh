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
Usage: build.sh <config> [options] [-- EXTRA_CMAKE_OPTIONS...]

Configs:
  cpu-debug       clang-18, Debug, logging, clang-tidy
  cpu-release     clang-18, Release, logging
  cpu-codecov     gcc-13, code coverage
  cpu-asan        clang-18, Debug, ASAN + UBSAN
  gpu-debug       clang-22, Debug, AMDGPU, logging, clang-tidy
  gpu-release     clang-22, Release, AMDGPU, logging
  gpu-asan        clang-18, Debug, AMDGPU, ASAN + UBSAN

Options:
  --source-dir DIR       Source directory (default: REPO_ROOT)
  --build-dir DIR        Build directory (default: build/)
  --iree-source-dir DIR  IREE source directory
  --target TARGET        Build target (default: all)

Extra cmake options can be passed after '--'.
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

CONFIG="$1"
shift

SOURCE_DIR="${REPO_ROOT}"
BUILD_DIR="build"
IREE_SOURCE_DIR=""
TARGET="all"
EXTRA_CMAKE_OPTIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --iree-source-dir)
      IREE_SOURCE_DIR="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_CMAKE_OPTIONS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

CMAKE_OPTIONS=(
  -GNinja
  "-S${SOURCE_DIR}"
  "-B${BUILD_DIR}"
)

if [[ -n "${IREE_SOURCE_DIR}" ]]; then
  CMAKE_OPTIONS+=("-DIREE_SOURCE_DIR=${IREE_SOURCE_DIR}")
fi

case "${CONFIG}" in
  cpu-debug)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-18
      -DCMAKE_CXX_COMPILER=clang++-18
      -DCMAKE_BUILD_TYPE=Debug
      -DFUSILLI_ENABLE_LOGGING=ON
      -DFUSILLI_ENABLE_CLANG_TIDY=ON
    )
    ;;
  cpu-release)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-18
      -DCMAKE_CXX_COMPILER=clang++-18
      -DCMAKE_BUILD_TYPE=Release
      -DFUSILLI_ENABLE_LOGGING=ON
    )
    ;;
  cpu-codecov)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=gcc-13
      -DCMAKE_CXX_COMPILER=g++-13
      -DFUSILLI_CODE_COVERAGE=ON
    )
    ;;
  cpu-asan)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-18
      -DCMAKE_CXX_COMPILER=clang++-18
      -DCMAKE_BUILD_TYPE=Debug
      -DFUSILLI_ENABLE_ASAN=ON
      -DFUSILLI_ENABLE_UBSAN=ON
    )
    ;;
  gpu-debug)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-22
      -DCMAKE_CXX_COMPILER=clang++-22
      -DCMAKE_BUILD_TYPE=Debug
      -DFUSILLI_SYSTEMS_AMDGPU=ON
      -DFUSILLI_ENABLE_LOGGING=ON
      -DFUSILLI_ENABLE_CLANG_TIDY=ON
    )
    ;;
  gpu-release)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-22
      -DCMAKE_CXX_COMPILER=clang++-22
      -DCMAKE_BUILD_TYPE=Release
      -DFUSILLI_SYSTEMS_AMDGPU=ON
      -DFUSILLI_ENABLE_LOGGING=ON
    )
    ;;
  gpu-asan)
    CMAKE_OPTIONS+=(
      -DCMAKE_C_COMPILER=clang-18
      -DCMAKE_CXX_COMPILER=clang++-18
      -DCMAKE_BUILD_TYPE=Debug
      -DFUSILLI_SYSTEMS_AMDGPU=ON
      -DFUSILLI_ENABLE_ASAN=ON
      -DFUSILLI_ENABLE_UBSAN=ON
    )
    ;;
  *)
    echo "Unknown config: ${CONFIG}"
    usage
    ;;
esac

CMAKE_OPTIONS+=("${EXTRA_CMAKE_OPTIONS[@]+"${EXTRA_CMAKE_OPTIONS[@]}"}")

echo "=== Fusilli build: config=${CONFIG} ==="
echo "=== CMake options: ${CMAKE_OPTIONS[*]} ==="

cmake "${CMAKE_OPTIONS[@]}"
cmake --build "${BUILD_DIR}" --target "${TARGET}"
