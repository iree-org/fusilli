// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[2,3,224,224],f32>, %arg0: !torch.vtensor<[2,3,224,224],f32>, %arg1: !torch.vtensor<[1,3,1,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_IN_0_val_0_pointwise_div = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_0_val_1_pointwise_div = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_0_val_2_pointwise_div = torch.constant.int 2
// TORCH-CHECK:       %permute_IN_0_val_3_pointwise_div = torch.constant.int 3
// TORCH-CHECK:       %permute_IN_0_pointwise_div = torch.prim.ListConstruct %permute_IN_0_val_0_pointwise_div, %permute_IN_0_val_1_pointwise_div, %permute_IN_0_val_2_pointwise_div, %permute_IN_0_val_3_pointwise_div : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_in0_pointwise_div_perm = torch.aten.permute %arg0, %permute_IN_0_pointwise_div : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int> -> !torch.vtensor<[2,3,224,224],f32>
// TORCH-CHECK:       %permute_IN_1_val_0_pointwise_div = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_1_val_1_pointwise_div = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_1_val_2_pointwise_div = torch.constant.int 2
// TORCH-CHECK:       %permute_IN_1_val_3_pointwise_div = torch.constant.int 3
// TORCH-CHECK:       %permute_IN_1_pointwise_div = torch.prim.ListConstruct %permute_IN_1_val_0_pointwise_div, %permute_IN_1_val_1_pointwise_div, %permute_IN_1_val_2_pointwise_div, %permute_IN_1_val_3_pointwise_div : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_in1_pointwise_div_perm = torch.aten.permute %arg1, %permute_IN_1_pointwise_div : !torch.vtensor<[1,3,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,3,1,1],f32>
// TORCH-CHECK:       %result_perm = torch.aten.div.Tensor %arg0_in0_pointwise_div_perm, %arg1_in1_pointwise_div_perm : !torch.vtensor<[2,3,224,224],f32>, !torch.vtensor<[1,3,1,1],f32> -> !torch.vtensor<[2,3,224,224],f32>
// TORCH-CHECK:       %permute_OUT_0_val_0_pointwise_div = torch.constant.int 0
// TORCH-CHECK:       %permute_OUT_0_val_1_pointwise_div = torch.constant.int 1
// TORCH-CHECK:       %permute_OUT_0_val_2_pointwise_div = torch.constant.int 2
// TORCH-CHECK:       %permute_OUT_0_val_3_pointwise_div = torch.constant.int 3
// TORCH-CHECK:       %permute_OUT_0_pointwise_div = torch.prim.ListConstruct %permute_OUT_0_val_0_pointwise_div, %permute_OUT_0_val_1_pointwise_div, %permute_OUT_0_val_2_pointwise_div, %permute_OUT_0_val_3_pointwise_div : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_OUT_0_pointwise_div : !torch.vtensor<[2,3,224,224],f32>, !torch.list<int> -> !torch.vtensor<[2,3,224,224],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[2,3,224,224],f32>, !torch.tensor<[2,3,224,224],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include "test_utils.h"
#include <fusilli.h>

#include <iostream>
#include <string>

using namespace fusilli;

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testBinaryPointwiseAsmEmitter(
      "pointwise_asm_emitter_div", "pointwise_div", mode,
      PointwiseAttr::Mode::DIV, {2, 3, 224, 224}, {1, 3, 1, 1});
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
