# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shared SDPA test utilities.
add_library(libsdpautils OBJECT sdpa_utils.cpp)
target_include_directories(libsdpautils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libsdpautils PUBLIC
  libfusilli
  libutils
  Catch2::Catch2WithMain
)
fusilli_enable_clang_tidy(libsdpautils)

add_fusilli_samples(
  PREFIX fusilli_conv_samples
  SRCS
    convolution/conv_fprop_nchw_kcrs.cpp
    convolution/conv_fprop_nchw_kcrs_grouped.cpp
    convolution/conv_fprop_nhwc_krsc.cpp
    convolution/conv_fprop_nhwc_krsc_with_pad.cpp
    convolution/conv_fprop_with_bias.cpp
    convolution/conv_fprop_with_relu.cpp
    convolution/conv_fprop_with_relu_and_bias.cpp
    convolution/conv_fprop_grouped_with_relu_and_bias.cpp
    convolution/conv_wgrad_nhwc_krsc.cpp
    convolution/conv_wgrad_nhwc_krsc_grouped.cpp
    convolution/conv_wgrad_nhwc_krsc_grouped_strided.cpp
    convolution/conv_dgrad_nhwc_krsc.cpp
    convolution/conv_dgrad_nhwc_krsc_grouped.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_pointwise_samples
  SRCS
    pointwise/pointwise_binary_bwd_ops.cpp
    pointwise/pointwise_add_transposed.cpp
    pointwise/pointwise_binary_ops.cpp
    pointwise/pointwise_binary_cmp_ops.cpp
    pointwise/pointwise_bwd.cpp
    pointwise/pointwise_scalar_mul.cpp
    pointwise/pointwise_ternary_ops.cpp
    pointwise/pointwise_unary_ops.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_matmul_samples
  SRCS
    matmul/matmul_basic.cpp
    matmul/matmul_basic_with_bias.cpp
    matmul/matmul_batched.cpp
    matmul/matmul_batched_with_bias.cpp
    matmul/matmul_int4_fp16.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_layernorm_samples
  SRCS
    layernorm/layernorm_infer_nchw.cpp
    layernorm/layernorm_infer_nchw_scale_bias.cpp
    layernorm/layernorm_infer_nhwc_scale_bias.cpp
    layernorm/layernorm_train_nchw.cpp
    layernorm/layernorm_train_nchw_scale_bias.cpp
    layernorm/layernorm_train_nhwc_scale_bias.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_batchnorm_samples
  SRCS
    batchnorm/batchnorm_infer_nchw.cpp
    batchnorm/batchnorm_infer_nchw_scale_bias.cpp
    batchnorm/batchnorm_infer_nhwc_scale_bias.cpp
    # TODO(iree-org/iree#24008): batchnorm training fails to distribute.
    # batchnorm/batchnorm_train_nchw.cpp
    # batchnorm/batchnorm_train_nchw_scale_bias.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_rmsnorm_samples
  SRCS
    rmsnorm/rmsnorm_infer_nchw.cpp
    rmsnorm/rmsnorm_infer_nchw_scale.cpp
    rmsnorm/rmsnorm_infer_nhwc_scale.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_reduction_samples
  SRCS
    reduction/reduction_ops.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_custom_op_samples
  SRCS
    custom_op/custom_op_add.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)



add_fusilli_samples(
  PREFIX fusilli_sdpa_custom_op_samples
  SRCS
    custom_op/sdpa/sdpa_fprop_basic_mha.cpp
    custom_op/sdpa/sdpa_fprop_causal.cpp
    custom_op/sdpa/sdpa_fprop_custom_scale.cpp
    custom_op/sdpa/sdpa_fprop_with_mask.cpp
    # TODO(#277): torch-mlir + IREE lowering needs to support dropout.
    # custom_op/sdpa/sdpa_fprop_dropout.cpp
    custom_op/sdpa/sdpa_fprop_gqa.cpp
    custom_op/sdpa/sdpa_fprop_gqa_hk_ne_hv.cpp
    custom_op/sdpa/sdpa_fprop_cross_attn.cpp
  DEPS
    libfusilli
    libutils
    libsdpautils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_sdpa_samples
  SRCS
    sdpa/sdpa_fprop_basic_mha.cpp
    sdpa/sdpa_fprop_causal.cpp
    sdpa/sdpa_fprop_custom_scale.cpp
    sdpa/sdpa_fprop_with_mask.cpp
    # TODO(#277): torch-mlir + IREE lowering needs to support dropout.
    # sdpa/sdpa_fprop_dropout.cpp
    sdpa/sdpa_fprop_gqa.cpp
    sdpa/sdpa_fprop_gqa_hk_ne_hv.cpp
    sdpa/sdpa_fprop_cross_attn.cpp
  DEPS
    libfusilli
    libutils
    libsdpautils
    Catch2::Catch2WithMain
)
