# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find lit program
fusilli_find_program(lit INSTALL_INSTRUCTIONS
                     "Please install lit (e.g., pip install lit).")

# Find iree-opt program
fusilli_find_program(
  iree-opt INSTALL_INSTRUCTIONS
  "Please install iree-opt (e.g., pip install iree-base-compiler).")

# Find FileCheck program
fusilli_find_program(FileCheck INSTALL_INSTRUCTIONS "Please install FileCheck.")

# Enable HIP tests if AMD GPU is enabled (ensures GPU detected) and HIP is found
if(FUSILLI_SYSTEMS_AMDGPU)
  set(HIP_PLATFORM "amd")
  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "AMD GPU detected and HIP found: Enabling HIP tests")
    add_subdirectory(hip_tests)
  else()
    message(STATUS "HIP not found: Skipping HIP tests")
  endif()
endif()

add_fusilli_tests(
  PREFIX
    fusilli_attribute_tests
  SRCS
    test_attributes.cpp
    test_tensor_attributes.cpp
    test_conv_attributes.cpp
    test_layernorm_attributes.cpp
    test_matmul_attributes.cpp
    test_pointwise_attributes.cpp
    test_reduction_attributes.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain)

add_fusilli_tests(
  PREFIX
    fusilli_node_tests
  SRCS
    test_conv_node.cpp
    test_layernorm_node.cpp
    test_matmul_node.cpp
    test_pointwise_node.cpp
    test_reduction_node.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain)

add_fusilli_tests(
  PREFIX
    fusilli_graph_tests
  SRCS
    test_graph.cpp
    test_context.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain)

add_fusilli_tests(
  PREFIX
    fusilli_backend_tests
  SRCS
    test_backend.cpp
    test_buffer.cpp
    test_compile_command.cpp
    test_compile_session.cpp
    test_handle.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain)

add_fusilli_tests(
  PREFIX
    fusilli_support_tests
  SRCS
    test_cache.cpp
    test_dllib.cpp
    test_float_types.cpp
    test_memstream.cpp
    test_ssa_validation.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain)

add_fusilli_tests(
  PREFIX
    fusilli_utils_tests
  SRCS
    test_utils.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain)

# Keep test_logging.cpp a separate test target as we manipulate logging env vars
# here which can suppress logs from other tests.
add_fusilli_tests(
  PREFIX
    fusilli_logging_tests
  SRCS
    test_logging.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain)

add_fusilli_lit_tests(
  SRCS
    lit/test_asm_emitter.cpp
    lit/test_conv_asm_emitter_nchw_kcrs.cpp
    lit/test_conv_asm_emitter_nchw_krsc.cpp
    lit/test_conv_asm_emitter_nhwc_krsc.cpp
    lit/test_conv_asm_emitter_nhwc_kcrs.cpp
    lit/test_conv_asm_emitter_nchw_kcrs_with_pad.cpp
    lit/test_conv_asm_emitter_nhwc_krsc_with_pad.cpp
    lit/test_conv_asm_emitter_nhwc_krsc_with_relu.cpp
    lit/test_conv_asm_emitter_ndhwc_kdrsc.cpp
    lit/test_conv_asm_emitter_ndhwc_kdrsc_grouped.cpp
    lit/test_conv_asm_emitter_nhwc_krsc_grouped.cpp
    lit/test_pointwise_asm_emitter_relu.cpp
    lit/test_pointwise_asm_emitter_add.cpp
    lit/test_pointwise_asm_emitter_add_transposed.cpp
    lit/test_pointwise_asm_emitter_ceil.cpp
    lit/test_pointwise_asm_emitter_cmp_eq.cpp
    lit/test_pointwise_asm_emitter_cmp_neq.cpp
    lit/test_pointwise_asm_emitter_cmp_gt.cpp
    lit/test_pointwise_asm_emitter_cmp_ge.cpp
    lit/test_pointwise_asm_emitter_cmp_lt.cpp
    lit/test_pointwise_asm_emitter_cmp_le.cpp
    lit/test_pointwise_asm_emitter_div.cpp
    lit/test_pointwise_asm_emitter_mul.cpp
    lit/test_pointwise_asm_emitter_sigmoid.cpp
    lit/test_pointwise_asm_emitter_tanh.cpp
    lit/test_pointwise_asm_emitter_sub.cpp
    lit/test_conv_wgrad_asm_emitter_nhwc_krsc.cpp
    lit/test_conv_wgrad_asm_emitter_nhwc_krsc_grouped.cpp
    lit/test_conv_wgrad_asm_emitter_nhwc_krsc_grouped_strided.cpp
    lit/test_conv_dgrad_asm_emitter_nhwc_kcrs.cpp
    lit/test_conv_dgrad_asm_emitter_nhwc_kcrs_grouped.cpp
    lit/test_layernorm_infer_asm_emitter_nchw.cpp
    lit/test_layernorm_infer_asm_emitter_scale_bias_nhwc.cpp
    lit/test_layernorm_train_asm_emitter_nchw.cpp
    lit/test_layernorm_train_asm_emitter_scale_bias_nhwc.cpp
    lit/test_matmul_asm_emitter_basic.cpp
    lit/test_matmul_asm_emitter_batched.cpp
    lit/test_matmul_asm_emitter_broadcast_3D.cpp
    lit/test_matmul_asm_emitter_broadcast_4D.cpp
    lit/test_matmul_asm_emitter_noncontiguous.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2
  TOOLS
    FileCheck
    iree-opt
    iree-compile)
