// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies GQA SDPA built-in op ASM emission:
//   - Q has 8 heads, KV has 2 heads (4:1 ratio)
//   - K/V heads are repeated before conversion to builtin tensors

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func @main(
// CHECK-SAME:      %k: !torch.vtensor<[1,2,64,64],f16>
// CHECK-SAME:      %q: !torch.vtensor<[1,8,64,64],f16>
// CHECK-SAME:      %v: !torch.vtensor<[1,2,64,64],f16>
// CHECK:           %k_gqa_sdpa = torch.aten.repeat_interleave.self_int %k_sdpa_perm, %gqa_repeats_sdpa, %gqa_dim_sdpa, %none_output_size_sdpa
// CHECK:           %v_gqa_sdpa = torch.aten.repeat_interleave.self_int %v_sdpa_perm, %gqa_repeats_sdpa, %gqa_dim_sdpa, %none_output_size_sdpa
// CHECK:           %k_tensor_sdpa = torch_c.to_builtin_tensor %k_gqa_sdpa
// CHECK:           %v_tensor_sdpa = torch_c.to_builtin_tensor %v_gqa_sdpa
// CHECK:           %online_attention_sdpa:3 = iree_linalg_ext.online_attention
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
  g.setName("sdpa_asm_emitter_gqa").setIODataType(DataType::Half);

  std::vector<int64_t> qDim = {1, 8, 64, 64};
  auto qStride =
      generateStrideFromDim(qDim, getContiguousStrideOrder(qDim.size()));

  std::vector<int64_t> kvDim = {1, 2, 64, 64};
  auto kvStride =
      generateStrideFromDim(kvDim, getContiguousStrideOrder(kvDim.size()));

  auto q = g.tensor(
      TensorAttr().setName("q").setDim(qDim).setStride(qStride).setDataType(
          DataType::Half));
  auto k = g.tensor(
      TensorAttr().setName("k").setDim(kvDim).setStride(kvStride).setDataType(
          DataType::Half));
  auto v = g.tensor(
      TensorAttr().setName("v").setDim(kvDim).setStride(kvStride).setDataType(
          DataType::Half));

  auto sdpaAttr = SdpaAttr().setName("sdpa").setEnableGqa(true);
  auto o = g.sdpa(q, k, v, /*mask=*/nullptr, sdpaAttr);

  std::vector<int64_t> oDim = {1, 8, 64, 64};
  auto oStride =
      generateStrideFromDim(oDim, getContiguousStrideOrder(oDim.size()));
  o->setDim(oDim)
      .setStride(oStride)
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
