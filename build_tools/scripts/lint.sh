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
Usage: lint.sh [options]

Options:
  --files FILE...      Run on specific files (default: all files)
EOF
  exit 1
}

ARGS=(--all-files)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --files)
      shift
      ARGS=(--files "$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

cd "${REPO_ROOT}"

echo "=== Fusilli lint ==="
pre-commit run "${ARGS[@]}"
