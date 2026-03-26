// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies SDPA custom op ASM emission (without attn_mask):
//   - Module-scope function definition with
//   torch.aten.scaled_dot_product_attention
//   - Scalar constants baked into the template (dropout_p, is_causal, scale,
//   enable_gqa)
//   - 3 tensor inputs (Q, K, V) with f16 dtype
//   - func.call with static-to-dynamic casts

// clang-format off
//
// Module-scope custom function definition:
// CHECK:       module @module {
// CHECK:         func.func private @sdpa(
// CHECK:             %arg0: !torch.vtensor<[1,8,64,64],f16>,
// CHECK:             %arg1: !torch.vtensor<[1,8,64,64],f16>,
// CHECK:             %arg2: !torch.vtensor<[1,8,64,64],f16>)
// CHECK:             -> !torch.vtensor<[1,8,64,64],f16> {
// CHECK:           %none_mask = torch.constant.none
// CHECK:           %dropout = torch.constant.float 0.000000e+00
// CHECK:           %is_causal = torch.constant.bool false
// CHECK:           %scale = torch.constant.none
// CHECK:           %enable_gqa = torch.constant.bool false
// CHECK:           %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
// CHECK:               %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
// CHECK:               !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>,
// CHECK:               !torch.vtensor<[1,8,64,64],f16>, !torch.none, !torch.float, !torch.bool,
// CHECK:               !torch.none, !torch.bool -> !torch.vtensor<[1,8,64,64],f16>
// CHECK:           return %0 : !torch.vtensor<[1,8,64,64],f16>
// CHECK:         }
//
// Main function with casts and call:
// CHECK:         func.func @main(
// CHECK-SAME:      %sdpa_OUT_0_: !torch.tensor<[1,8,64,64],f16>
// CHECK-SAME:      %k: !torch.vtensor<[1,8,64,64],f16>
// CHECK-SAME:      %q: !torch.vtensor<[1,8,64,64],f16>
// CHECK-SAME:      %v: !torch.vtensor<[1,8,64,64],f16>
// CHECK:           %sdpa_OUT_0_sdpa_perm = func.call @sdpa(%q_sdpa_i0_perm, %k_sdpa_i1_perm, %v_sdpa_i2_perm) : (!torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>) -> !torch.vtensor<[1,8,64,64],f16>
// CHECK:           torch.overwrite.tensor.contents %sdpa_OUT_0 overwrites %sdpa_OUT_0_
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("custom_op_asm_emitter_sdpa").setIODataType(DataType::Half);

  std::vector<int64_t> dim = {1, 8, 64, 64};
  auto stride =
      generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));

  auto q = g.tensor(
      TensorAttr().setName("q").setDim(dim).setStride(stride).setDataType(
          DataType::Half));
  auto k = g.tensor(
      TensorAttr().setName("k").setDim(dim).setStride(stride).setDataType(
          DataType::Half));
  auto v = g.tensor(
      TensorAttr().setName("v").setDim(dim).setStride(stride).setDataType(
          DataType::Half));

  // Inline MLIR template for SDPA (no attention mask, default scalar params).
  std::string sdpaMlir = R"mlir(
  func.func private @{FUNC_NAME}(
      %arg0: {IN0_TYPE},
      %arg1: {IN1_TYPE},
      %arg2: {IN2_TYPE})
      -> {OUT0_TYPE} {
    %none_mask = torch.constant.none
    %dropout = torch.constant.float 0.000000e+00
    %is_causal = torch.constant.bool false
    %scale = torch.constant.none
    %enable_gqa = torch.constant.bool false
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        {IN0_TYPE}, {IN1_TYPE},
        {IN2_TYPE}, !torch.none, !torch.float, !torch.bool,
        !torch.none, !torch.bool -> {OUT0_TYPE}
    return %0 : {OUT0_TYPE}
  }
)mlir";

  CustomOpAttr sdpaAttr;
  sdpaAttr.setName("sdpa").setMlir(sdpaMlir).setNumOutputs(1);

  auto outs = g.customOp({q, k, v}, sdpaAttr);
  outs[0]
      ->setDim(dim)
      .setStride(stride)
      .setDataType(DataType::Half)
      .setOutput(true);

  auto status = g.validate();
  if (isError(status)) {
    std::cerr << "Validation failed: " << status << std::endl;
    return 1;
  }

  auto asmOrErr = g.emitAsm();
  if (isError(asmOrErr)) {
    std::cerr << "ASM emission failed: " << asmOrErr << std::endl;
    return 1;
  }

  auto indentErr = checkMlirIndentation(*asmOrErr);
  if (isError(indentErr)) {
    std::cerr << "Indentation check failed: " << indentErr << std::endl;
    return 1;
  }

  std::cout << *asmOrErr << std::endl;
  return 0;
}
