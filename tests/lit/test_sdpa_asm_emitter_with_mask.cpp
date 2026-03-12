// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies SDPA with attention mask ASM emission:
//   - 4 tensor inputs (Q, K, V, mask)
//   - mask type is a vtensor (not none)

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func @main(
// CHECK-SAME:      %mask: !torch.vtensor<[1,1,64,64],f16>
// CHECK:           %dropout_sdpa = torch.constant.float 0.000000e+00
// CHECK:           %is_causal_sdpa = torch.constant.bool false
// CHECK:           %scale_sdpa = torch.constant.none
// CHECK:           %enable_gqa_sdpa = torch.constant.bool false
// CHECK:           torch.aten.scaled_dot_product_attention
// CHECK-NOT:       !torch.none, !torch.float
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
  g.setName("sdpa_asm_emitter_with_mask").setIODataType(DataType::Half);

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

  std::vector<int64_t> maskDim = {1, 1, 64, 64};
  auto maskStride =
      generateStrideFromDim(maskDim, getContiguousStrideOrder(maskDim.size()));
  auto mask = g.tensor(TensorAttr()
                           .setName("mask")
                           .setDim(maskDim)
                           .setStride(maskStride)
                           .setDataType(DataType::Half));

  auto sdpaAttr = SdpaAttr().setName("sdpa");
  auto o = g.sdpa(q, k, v, mask, sdpaAttr);
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
