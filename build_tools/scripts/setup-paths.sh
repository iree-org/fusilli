#! /usr/bin/bash

if [ -z "$FUSILLI_DIR" ]; then
    echo "FUSILLI_DIR is unset or empty."
    exit 1
fi

if [ -z "$IREE_DIR" ]; then
    echo "IREE_DIR is unset or empty."
    exit 1
fi

if [ -z "$HIP_DIR" ]; then
    echo "HIP_DIR is unset or empty."
    exit 1
fi


if [ -z "$HIPDNN_DIR" ]; then
    echo "HIPDNN_DIR is unset or empty."
    exit 1
fi

export FUSILLI_TEST_BUILD_DIR=${FUSILLI_DIR}/build-fusilli-test
export FUSILLI_LIB_BUILD_DIR=${FUSILLI_DIR}/build-fusilli-lib
export FUSILLI_PLUGIN_BUILD_DIR=${FUSILLI_DIR}/build-fusilli-plugin
export HIPDNN_BUILD_DIR=${HIPDNN_DIR}/build-hipdnn

echo "FUSILLI_DIR: $FUSILLI_DIR"
echo "IREE_DIR: $IREE_DIR"
echo "HIP_DIR: $HIP_DIR"
echo "HIPDNN_DIR: $HIPDNN_DIR"
