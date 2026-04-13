// Copyright 2026 Advanced Micro Devices, Inc.
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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,256,64,32],f32>, %arg0: !torch.vtensor<[16,256,64,32],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_IN_0_val_0_pointwise_identity = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_0_val_1_pointwise_identity = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_0_val_2_pointwise_identity = torch.constant.int 2
// TORCH-CHECK:       %permute_IN_0_val_3_pointwise_identity = torch.constant.int 3
// TORCH-CHECK:       %permute_IN_0_pointwise_identity = torch.prim.ListConstruct %permute_IN_0_val_0_pointwise_identity, %permute_IN_0_val_1_pointwise_identity, %permute_IN_0_val_2_pointwise_identity, %permute_IN_0_val_3_pointwise_identity : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_pointwise_identity_perm = torch.aten.permute %arg0, %permute_IN_0_pointwise_identity : !torch.vtensor<[16,256,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %none_pointwise_identity = torch.constant.none
// TORCH-CHECK:       %result_pointwise_identity_perm = torch.aten.clone %arg0_pointwise_identity_perm, %none_pointwise_identity : !torch.vtensor<[16,256,64,32],f32>, !torch.none -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %permute_OUT_0_val_0_pointwise_identity = torch.constant.int 0
// TORCH-CHECK:       %permute_OUT_0_val_1_pointwise_identity = torch.constant.int 1
// TORCH-CHECK:       %permute_OUT_0_val_2_pointwise_identity = torch.constant.int 2
// TORCH-CHECK:       %permute_OUT_0_val_3_pointwise_identity = torch.constant.int 3
// TORCH-CHECK:       %permute_OUT_0_pointwise_identity = torch.prim.ListConstruct %permute_OUT_0_val_0_pointwise_identity, %permute_OUT_0_val_1_pointwise_identity, %permute_OUT_0_val_2_pointwise_identity, %permute_OUT_0_val_3_pointwise_identity : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_pointwise_identity_perm, %permute_OUT_0_pointwise_identity : !torch.vtensor<[16,256,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,256,64,32],f32>, !torch.tensor<[16,256,64,32],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "transient-memory-size": 0
// AMDGPU-STATS-CHECK: "dispatch-count": 0
// CPU-STATS-CHECK: "transient-memory-size": 0
// CPU-STATS-CHECK: "dispatch-count": 0
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <iostream>
#include <string>

using namespace fusilli;

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testUnaryPointwiseAsmEmitter(
      "pointwise_asm_emitter_identity", "pointwise_identity", mode,
      PointwiseAttr::Mode::IDENTITY, {16, 256, 64, 32});
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
