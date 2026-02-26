#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/setup-paths.sh

if [ -z "$FUSILLI_TEST_BUILD_DIR" ]; then
    echo "FUSILLI_TEST_BUILD_DIR is unset or empty."
    exit 1
fi

ctest --test-dir ${FUSILLI_TEST_BUILD_DIR}/ \
    --output-on-failure \
    --extra-verbose \
    --timeout 120 \
    -j $(nproc)
