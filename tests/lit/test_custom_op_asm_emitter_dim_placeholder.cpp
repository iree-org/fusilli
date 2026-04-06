// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @my_add(
// CHECK-NEXT:      %arg0: !torch.vtensor<[4,8],f32>,
// CHECK-NEXT:      %arg1: !torch.vtensor<[4,8],f32>)
// CHECK-NEXT:      -> !torch.vtensor<[4,8],f32> {
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
// CHECK:               !torch.vtensor<[4,8],f32>
// CHECK:               -> !torch.vtensor<[4,8],f32>
// CHECK:           return %0 : !torch.vtensor<[4,8],f32>
// CHECK:         }
// CHECK:         func.func @main(
// CHECK:           func.call @my_add
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <iostream>
#include <string>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("custom_op_asm_emitter_dim_placeholder")
      .setIODataType(DataType::Float);

  auto a = g.tensor(
      TensorAttr().setName("a").setDim({4, 8}).setStride({8, 1}).setDataType(
          DataType::Float));
  auto b = g.tensor(
      TensorAttr().setName("b").setDim({4, 8}).setStride({8, 1}).setDataType(
          DataType::Float));

  // Composes tensor types from individual {IN0_DIM0}/{IN0_DIM1} placeholders
  // instead of using {IN0_TYPE}.  Shows that users can build type strings
  // from dimension values when they need custom type compositions.
  std::string addMlir = R"(
  func.func private @{FUNC_NAME}(
      %arg0: !torch.vtensor<[{IN0_DIM0},{IN0_DIM1}],{IN0_DTYPE}>,
      %arg1: !torch.vtensor<[{IN1_DIM0},{IN1_DIM1}],{IN1_DTYPE}>)
      -> !torch.vtensor<[{OUT0_DIM0},{OUT0_DIM1}],{OUT0_DTYPE}> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1}],{IN0_DTYPE}>,
          !torch.vtensor<[{IN1_DIM0},{IN1_DIM1}],{IN1_DTYPE}>, !torch.int
        -> !torch.vtensor<[{OUT0_DIM0},{OUT0_DIM1}],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[{OUT0_DIM0},{OUT0_DIM1}],{OUT0_DTYPE}>
  }
)";

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(addMlir).setNumOutputs(1);

  auto outs = g.customOp({a, b}, addAttr);
  outs[0]
      ->setDim({4, 8})
      .setStride({8, 1})
      .setDataType(DataType::Float)
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
