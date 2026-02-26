#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/setup-paths.sh

mkdir -p ${FUSILLI_TEST_BUILD_DIR}
cmake -G Ninja -B ${FUSILLI_TEST_BUILD_DIR} \
    -S ${FUSILLI_DIR} \
	-DFUSILLI_BUILD_TESTS=ON \
	-DFUSILLI_BUILD_BENCHMARKS=ON \
	-DIREE_SOURCE_DIR=$IREE_DIR \
	-DHIP_DIR=${HIP_DIR}/lib/cmake \
	-DHIP_PLATFORM=amd \
	-DIREE_USE_SYSTEM_DEPS=ON \
    ${FUSILLI_CMAKE_FLAGS}

cmake --build ${FUSILLI_TEST_BUILD_DIR}/
