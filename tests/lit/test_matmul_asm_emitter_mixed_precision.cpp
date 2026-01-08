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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[64,256],f32>, %arg0_matrix_a: !torch.vtensor<[64,128],f32>, %arg1_matrix_b: !torch.vtensor<[128,256],bf16>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_A_val_0_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_A_val_1_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_A_matmul = torch.prim.ListConstruct %permute_A_val_0_matmul, %permute_A_val_1_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_matrix_a_perm = torch.aten.permute %arg0_matrix_a, %permute_A_matmul : !torch.vtensor<[64,128],f32>, !torch.list<int> -> !torch.vtensor<[64,128],f32>
// TORCH-CHECK:       %permute_B_val_0_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_B_val_1_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_B_matmul = torch.prim.ListConstruct %permute_B_val_0_matmul, %permute_B_val_1_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_matrix_b_perm = torch.aten.permute %arg1_matrix_b, %permute_B_matmul : !torch.vtensor<[128,256],bf16>, !torch.list<int> -> !torch.vtensor<[128,256],bf16>
// TORCH-CHECK:       %dtype_B_cast_matmul = torch.constant.int 6
// TORCH-CHECK:       %false_B_matmul = torch.constant.bool false
// TORCH-CHECK:       %none_B_matmul = torch.constant.none
// TORCH-CHECK:       %arg1_matrix_b_perm_cast = torch.aten.to.dtype %arg1_matrix_b_perm, %dtype_B_cast_matmul, %false_B_matmul, %false_B_matmul, %none_B_matmul : !torch.vtensor<[128,256],bf16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       %result_perm = torch.aten.matmul %arg0_matrix_a_perm, %arg1_matrix_b_perm_cast : !torch.vtensor<[64,128],f32>, !torch.vtensor<[128,256],f32> -> !torch.vtensor<[64,256],f32>
// TORCH-CHECK:       %permute_C_val_0_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_C_val_1_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_C_matmul = torch.prim.ListConstruct %permute_C_val_0_matmul, %permute_C_val_1_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_C_matmul : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[64,256],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[64,256],f32>, !torch.tensor<[64,256],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testMatmulAsmEmitterMixedPrecision(const std::string &mode) {
  int64_t m = 64, k = 128, n = 256;
  auto graph = std::make_shared<Graph>();
  graph->setName("matmul_asm_emitter_mixed_precision");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto aT = graph->tensor(TensorAttr()
                              .setName("arg0_matrix_a")
                              .setDim({m, k})
                              .setStride({k, 1})
                              .setDataType(DataType::Float));

  auto bT = graph->tensor(TensorAttr()
                              .setName("arg1_matrix_b")
                              .setDim({k, n})
                              .setStride({n, 1})
                              .setDataType(DataType::BFloat16));

  auto matmulAttr = MatmulAttr().setName("matmul");

  auto cT = graph->matmul(aT, bT, matmulAttr);

  cT->setName("result").setOutput(true).setDataType(DataType::Float);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testMatmulAsmEmitterMixedPrecision(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
