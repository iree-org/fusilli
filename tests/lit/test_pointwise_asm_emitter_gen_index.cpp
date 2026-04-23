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
// TORCH-CHECK:       %gen_index_empty_pointwise_gen_index = tensor.empty() : tensor<16x256x64x32xf32>
// TORCH-CHECK:       %gen_index_linalg_pointwise_gen_index = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%gen_index_empty_pointwise_gen_index : tensor<16x256x64x32xf32>) {
// TORCH-CHECK:       ^bb0(%gen_index_out_pointwise_gen_index: f32):
// TORCH-CHECK:         %gen_index_idx_pointwise_gen_index = linalg.index 2 : index
// TORCH-CHECK:         %gen_index_int_pointwise_gen_index = arith.index_cast %gen_index_idx_pointwise_gen_index : index to i64
// TORCH-CHECK:         %gen_index_val_pointwise_gen_index = arith.sitofp %gen_index_int_pointwise_gen_index : i64 to f32
// TORCH-CHECK:         linalg.yield %gen_index_val_pointwise_gen_index : f32
// TORCH-CHECK:       } -> tensor<16x256x64x32xf32>
// TORCH-CHECK:       %result_pointwise_gen_index_perm = torch_c.from_builtin_tensor %gen_index_linalg_pointwise_gen_index : tensor<16x256x64x32xf32> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %result = torch.aten.permute %result_pointwise_gen_index_perm, %permute_OUT_0_pointwise_gen_index : !torch.vtensor<[16,256,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,256,64,32],f32>, !torch.tensor<[16,256,64,32],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "transient-memory-size": 0
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "transient-memory-size": 0
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include <fusilli.h>

#include "pointwise_utils.h"

#include <iostream>
#include <string>

using namespace fusilli;

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testGenIndexAsmEmitter("pointwise_asm_emitter_gen_index",
                                       "pointwise_gen_index", mode,
                                       {16, 256, 64, 32}, /*axis=*/2);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
