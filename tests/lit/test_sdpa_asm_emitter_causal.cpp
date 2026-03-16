// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies causal SDPA built-in op ASM emission:
//   - is_causal set to true

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func @main(
// CHECK:           %is_causal_sdpa = torch.constant.bool true
// CHECK:           torch.aten.scaled_dot_product_attention
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
  g.setName("sdpa_asm_emitter_causal").setIODataType(DataType::Half);

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

  auto sdpaAttr = SdpaAttr().setName("sdpa").setIsCausal(true);
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
