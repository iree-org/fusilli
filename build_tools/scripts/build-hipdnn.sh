#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/setup-paths.sh

mkdir -p ${HIPDNN_BUILD_DIR}
cmake -G Ninja \
    -B ${HIPDNN_BUILD_DIR} \
    -S ${HIPDNN_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DHIP_PLATFORM=amd \
    -DHIP_DNN_BUILD_PLUGINS=OFF \
    -DHIP_DNN_GENERATE_SDK_HEADERS=OFF \
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_CLANG_FORMAT=OFF \
	-DHIP_PLATFORM=amd \
	-DHIP_DIR=${HIP_DIR}/lib/cmake \
	-DIREE_USE_SYSTEM_DEPS=ON \
    -DHIPDNN_SKIP_TESTS=ON \

cmake --build ${HIPDNN_BUILD_DIR}/
