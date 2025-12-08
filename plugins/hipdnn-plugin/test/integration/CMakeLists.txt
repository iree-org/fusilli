# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Threads REQUIRED)

# Integration tests
add_fusilli_plugin_test(
  NAME fusilli_plugin_integration_test
  SRCS
    test_convfprop.cpp
    test_basic.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
    FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
    FUSILLI_PLUGIN_ENGINE_ID="${FUSILLI_PLUGIN_ENGINE_ID}"
)
