// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies basic SDPA built-in op ASM emission:
//   - Inputs stay in Torch value tensors at the graph boundary
//   - Q/K/V are bridged to builtin tensors for iree_linalg_ext.online_attention
//   - The online attention accumulator is normalized with Torch unsqueeze+div
//   - Default scale is materialized as 1/sqrt(head_dim)

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func @main(
// CHECK-SAME:      %sdpa_O_: !torch.tensor<[1,8,64,64],f16>
// CHECK-SAME:      %k: !torch.vtensor<[1,8,64,64],f16>
// CHECK-SAME:      %q: !torch.vtensor<[1,8,64,64],f16>
// CHECK-SAME:      %v: !torch.vtensor<[1,8,64,64],f16>
// CHECK:           %q_tensor_sdpa = torch_c.to_builtin_tensor %q_sdpa_perm
// CHECK:           %k_tensor_sdpa = torch_c.to_builtin_tensor %k_sdpa_perm
// CHECK:           %v_tensor_sdpa = torch_c.to_builtin_tensor %v_sdpa_perm
// CHECK:           %scale_sdpa = arith.constant 1.250000e-01 : f16
// CHECK:           %online_attention_sdpa:3 = iree_linalg_ext.online_attention
// CHECK:           %accum_sdpa = torch_c.from_builtin_tensor %online_attention_sdpa#0
// CHECK:           %sum_sdpa = torch_c.from_builtin_tensor %online_attention_sdpa#2
// CHECK:           %sum_expanded_sdpa = torch.aten.unsqueeze %sum_sdpa, %unsqueeze_dim_sdpa
// CHECK:           %sdpa_O_sdpa_perm = torch.aten.div.Tensor %accum_sdpa, %sum_expanded_sdpa
// CHECK:           torch.overwrite.tensor.contents
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
  g.setName("sdpa_asm_emitter_basic").setIODataType(DataType::Half);

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

  auto sdpaAttr = SdpaAttr().setName("sdpa");
  auto o = g.sdpa(q, k, v, /*mask=*/nullptr, sdpaAttr);
  o->setDim(dim).setStride(stride).setDataType(DataType::Half).setOutput(true);

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
