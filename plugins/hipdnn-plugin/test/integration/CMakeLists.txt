# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Threads REQUIRED)

# Integration tests
add_fusilli_plugin_test(
  NAME fusilli_plugin_test_basic
  SRCS test_basic.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
    FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
    FUSILLI_PLUGIN_ENGINE_ID="${FUSILLI_PLUGIN_ENGINE_ID}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_test_convfprop
  SRCS test_convfprop.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
    FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
    FUSILLI_PLUGIN_ENGINE_ID="${FUSILLI_PLUGIN_ENGINE_ID}"
)

add_fusilli_plugin_test(
  NAME fusilli_plugin_test_convfprop_fused
  SRCS test_convfprop_fused.cpp
  DEPS
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
  COMPILE_DEFS
    FUSILLI_PLUGIN_PATH="../${FUSILLI_PLUGIN_RELATIVE_PATH}"
    FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
    FUSILLI_PLUGIN_ENGINE_ID="${FUSILLI_PLUGIN_ENGINE_ID}"
)
