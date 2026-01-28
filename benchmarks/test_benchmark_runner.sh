#!/bin/bash
set -euo pipefail
set -x

# Arguments from CMake
BENCHMARK_RUNNER="$1"
BENCHMARK_DRIVER="$2"

# Use CLI backend for tests (--iree-codegen-tuning-spec-path not available via C API)
# TODO(iree-org/iree#23314): Remove this when tuning spec path support is added to C API
export FUSILLI_COMPILE_BACKEND_USE_CLI=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_COMMANDS="${SCRIPT_DIR}/test_commands.txt"
OUTPUT_CSV=$(mktemp)
python3 "${BENCHMARK_RUNNER}" \
  --commands-file "${TEST_COMMANDS}" \
  --csv "${OUTPUT_CSV}" \
  --driver "${BENCHMARK_DRIVER}" \
  --verbose
if [ ! -f "${OUTPUT_CSV}" ]; then
  echo "ERROR: Output CSV not created"
  exit 1
fi
# Count number of rows
NUM_ROWS=$(tail -n +2 "${OUTPUT_CSV}" | wc -l)
# Count non-empty, non-comment lines (matching Python script behavior)
EXPECTED_ROWS=$(grep -Ev '^\s*#|^\s*$' "${TEST_COMMANDS}" | wc -l)
if [ "${NUM_ROWS}" -ne "${EXPECTED_ROWS}" ]; then
  echo "ERROR: Expected ${EXPECTED_ROWS} rows, got ${NUM_ROWS}"
  exit 1
fi
# Using --iter 10, check column exists and has value 10
if ! grep -q "iter" "${OUTPUT_CSV}"; then
  echo "ERROR: 'iter' column not found in CSV"
  exit 1
fi
# Check that dispatch_count column exists
if ! grep -q "dispatch_count" "${OUTPUT_CSV}"; then
  echo "ERROR: 'dispatch_count' column not found in CSV"
  exit 1
fi
# Verify at least one row has iter=10
if ! tail -n +2 "${OUTPUT_CSV}" | cut -d',' -f6 | grep -q "10"; then
  echo "ERROR: Expected iter=10 not found"
  exit 1
fi

echo "PASSED: fusilli_benchmark_runner_tests (without tuning spec)"

# Test with tuning spec using --Xiree-compile flag
TEST_TUNING_SPEC="${SCRIPT_DIR}/test_tuning_spec.mlir"
OUTPUT_CSV_TUNED=$(mktemp)
python3 "${BENCHMARK_RUNNER}" \
  --commands-file "${TEST_COMMANDS}" \
  --csv "${OUTPUT_CSV_TUNED}" \
  --driver "${BENCHMARK_DRIVER}" \
  --Xiree-compile="--iree-codegen-tuning-spec-path=${TEST_TUNING_SPEC}" \
  --verbose
if [ ! -f "${OUTPUT_CSV_TUNED}" ]; then
  echo "ERROR: Output CSV (tuned) not created"
  exit 1
fi
# Count number of rows (should be same as non-tuned)
NUM_ROWS_TUNED=$(tail -n +2 "${OUTPUT_CSV_TUNED}" | wc -l)
if [ "${NUM_ROWS_TUNED}" -ne "${EXPECTED_ROWS}" ]; then
  echo "ERROR: Expected ${EXPECTED_ROWS} rows (tuned), got ${NUM_ROWS_TUNED}"
  exit 1
fi
echo "PASSED: fusilli_benchmark_runner_tests (with --Xiree-compile)"

# Test multiple --Xiree-compile flags
OUTPUT_CSV_MULTI=$(mktemp)
python3 "${BENCHMARK_RUNNER}" \
  --commands-file "${TEST_COMMANDS}" \
  --csv "${OUTPUT_CSV_MULTI}" \
  --driver "${BENCHMARK_DRIVER}" \
  --Xiree-compile="--iree-codegen-tuning-spec-path=${TEST_TUNING_SPEC}" \
  --Xiree-compile="--iree-opt-level=O3" \
  --verbose
if [ ! -f "${OUTPUT_CSV_MULTI}" ]; then
  echo "ERROR: Output CSV (multi flags) not created"
  exit 1
fi
# Count number of rows (should be same as non-tuned)
NUM_ROWS_MULTI=$(tail -n +2 "${OUTPUT_CSV_MULTI}" | wc -l)
if [ "${NUM_ROWS_MULTI}" -ne "${EXPECTED_ROWS}" ]; then
  echo "ERROR: Expected ${EXPECTED_ROWS} rows (multi flags), got ${NUM_ROWS_MULTI}"
  exit 1
fi
echo "PASSED: fusilli_benchmark_runner_tests (with multiple --Xiree-compile)"

rm -f "${OUTPUT_CSV}" "${OUTPUT_CSV_TUNED}" "${OUTPUT_CSV_MULTI}"
