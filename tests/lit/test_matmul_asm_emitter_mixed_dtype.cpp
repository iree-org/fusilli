// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} | iree-compile - --compile-to=input >/dev/null

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[64,256],f16>, %arg0_matrix_a: !torch.vtensor<[64,128],si4>, %arg1_matrix_b: !torch.vtensor<[128,256],f16>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_A_val_0_mixed_dtype_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_A_val_1_mixed_dtype_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_A_mixed_dtype_matmul = torch.prim.ListConstruct %permute_A_val_0_mixed_dtype_matmul, %permute_A_val_1_mixed_dtype_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_matrix_a_mixed_dtype_matmul_perm = torch.aten.permute %arg0_matrix_a, %permute_A_mixed_dtype_matmul : !torch.vtensor<[64,128],si4>, !torch.list<int> -> !torch.vtensor<[64,128],si4>
// TORCH-CHECK:       %permute_B_val_0_mixed_dtype_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_B_val_1_mixed_dtype_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_B_mixed_dtype_matmul = torch.prim.ListConstruct %permute_B_val_0_mixed_dtype_matmul, %permute_B_val_1_mixed_dtype_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_matrix_b_mixed_dtype_matmul_perm = torch.aten.permute %arg1_matrix_b, %permute_B_mixed_dtype_matmul : !torch.vtensor<[128,256],f16>, !torch.list<int> -> !torch.vtensor<[128,256],f16>
// TORCH-CHECK:       %dtype_A_cast_mixed_dtype_matmul = torch.constant.int 5
// TORCH-CHECK:       %non_blocking_A_cast_mixed_dtype_matmul = torch.constant.bool false
// TORCH-CHECK:       %copy_A_cast_mixed_dtype_matmul = torch.constant.bool false
// TORCH-CHECK:       %memory_format_A_cast_mixed_dtype_matmul = torch.constant.none
// TORCH-CHECK:       %arg0_matrix_a_mixed_dtype_matmul_perm_cast = torch.aten.to.dtype %arg0_matrix_a_mixed_dtype_matmul_perm, %dtype_A_cast_mixed_dtype_matmul, %non_blocking_A_cast_mixed_dtype_matmul, %copy_A_cast_mixed_dtype_matmul, %memory_format_A_cast_mixed_dtype_matmul : !torch.vtensor<[64,128],si4>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[64,128],f16>
// TORCH-CHECK:       %result_mixed_dtype_matmul_perm = torch.aten.matmul %arg0_matrix_a_mixed_dtype_matmul_perm_cast, %arg1_matrix_b_mixed_dtype_matmul_perm : !torch.vtensor<[64,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[64,256],f16>
// TORCH-CHECK:       %permute_C_val_0_mixed_dtype_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_C_val_1_mixed_dtype_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_C_mixed_dtype_matmul = torch.prim.ListConstruct %permute_C_val_0_mixed_dtype_matmul, %permute_C_val_1_mixed_dtype_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_mixed_dtype_matmul_perm, %permute_C_mixed_dtype_matmul : !torch.vtensor<[64,256],f16>, !torch.list<int> -> !torch.vtensor<[64,256],f16>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[64,256],f16>, !torch.tensor<[64,256],f16>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>

using namespace fusilli;

int main() {
  int64_t m = 64, k = 128, n = 256;
  auto graph = std::make_shared<Graph>();
  graph->setName("matmul_asm_emitter_mixed_dtype");
  graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto aT = graph->tensor(TensorAttr()
                              .setName("arg0_matrix_a")
                              .setDim({m, k})
                              .setStride({k, 1})
                              .setDataType(DataType::Int4));

  auto bT = graph->tensor(
      TensorAttr().setName("arg1_matrix_b").setDim({k, n}).setStride({n, 1}));

  auto matmulAttr = MatmulAttr().setName("mixed_dtype_matmul");
  auto cT = graph->matmul(aT, bT, matmulAttr);
  cT->setName("result").setOutput(true);

  auto status = graph->validate();
  if (isError(status)) {
    std::cerr << "Validation failed: " << status << std::endl;
    return 1;
  }

  auto asmOrErr = graph->emitAsm();
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
