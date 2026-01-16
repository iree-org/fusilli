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
// TORCH-CHECK:     func.func @main(%[[RESULT0:.+]]: !torch.tensor<[16,256,64,32],i1>, %[[ARG0:.+]]: !torch.vtensor<[16,256,64,32],f32>, %[[ARG1:.+]]: !torch.vtensor<[1,256,1,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %[[PERM0_0:.+]] = torch.constant.int 0
// TORCH-CHECK:       %[[PERM0_1:.+]] = torch.constant.int 1
// TORCH-CHECK:       %[[PERM0_2:.+]] = torch.constant.int 2
// TORCH-CHECK:       %[[PERM0_3:.+]] = torch.constant.int 3
// TORCH-CHECK:       %[[PERM0_LIST:.+]] = torch.prim.ListConstruct %[[PERM0_0]], %[[PERM0_1]], %[[PERM0_2]], %[[PERM0_3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %[[PERMUTE0:.+]] = torch.aten.permute %[[ARG0]], %[[PERM0_LIST]] : !torch.vtensor<[16,256,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %[[PERM1_0:.+]] = torch.constant.int 0
// TORCH-CHECK:       %[[PERM1_1:.+]] = torch.constant.int 1
// TORCH-CHECK:       %[[PERM1_2:.+]] = torch.constant.int 2
// TORCH-CHECK:       %[[PERM1_3:.+]] = torch.constant.int 3
// TORCH-CHECK:       %[[PERM1_LIST:.+]] = torch.prim.ListConstruct %[[PERM1_0]], %[[PERM1_1]], %[[PERM1_2]], %[[PERM1_3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %[[PERMUTE1:.+]] = torch.aten.permute %[[ARG1]], %[[PERM1_LIST]] : !torch.vtensor<[1,256,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,256,1,1],f32>
// TORCH-CHECK:       %[[CEIL:.+]] = torch.aten.ne.Tensor %[[PERMUTE0]], %[[PERMUTE1]] : !torch.vtensor<[16,256,64,32],f32>, !torch.vtensor<[1,256,1,1],f32> -> !torch.vtensor<[16,256,64,32],i1>
// TORCH-CHECK:       %[[PERM_OUT_0:.+]] = torch.constant.int 0
// TORCH-CHECK:       %[[PERM_OUT_1:.+]] = torch.constant.int 1
// TORCH-CHECK:       %[[PERM_OUT_2:.+]] = torch.constant.int 2
// TORCH-CHECK:       %[[PERM_OUT_3:.+]] = torch.constant.int 3
// TORCH-CHECK:       %[[PERM_OUT_LIST:.+]] = torch.prim.ListConstruct %[[PERM_OUT_0]], %[[PERM_OUT_1]], %[[PERM_OUT_2]], %[[PERM_OUT_3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %[[PERM_OUT:.+]] = torch.aten.permute %[[CEIL]], %[[PERM_OUT_LIST]] : !torch.vtensor<[16,256,64,32],i1>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],i1>
// TORCH-CHECK:       torch.overwrite.tensor.contents %[[PERM_OUT]] overwrites %[[RESULT0]] : !torch.vtensor<[16,256,64,32],i1>, !torch.tensor<[16,256,64,32],i1>
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
      "pointwise_asm_emitter_cmp_neq", "cmp_neq", mode,
      PointwiseAttr::Mode::CMP_NEQ, {16, 256, 64, 32}, {1, 256, 1, 1});
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
