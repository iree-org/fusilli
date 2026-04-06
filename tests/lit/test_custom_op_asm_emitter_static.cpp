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
// CHECK:         func.func private @my_add(%arg0: !torch.vtensor<[4],f32>,
// CHECK:                                    %arg1: !torch.vtensor<[4],f32>)
// CHECK:                                    -> !torch.vtensor<[4],f32> {
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
// CHECK:               : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int
// CHECK:               -> !torch.vtensor<[4],f32>
// CHECK:           return %0 : !torch.vtensor<[4],f32>
// CHECK:         }
//
// CHECK:         func.func @main(
// CHECK-SAME:      %my_add_OUT_0_: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f32>
// CHECK-SAME:      %b: !torch.vtensor<[4],f32>
// CHECK:           %my_add_OUT_0_my_add_perm = func.call @my_add(%a_my_add_i0_perm, %b_my_add_i1_perm) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
// CHECK:           torch.overwrite.tensor.contents %my_add_OUT_0 overwrites %my_add_OUT_0_
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
  g.setName("custom_op_asm_emitter_static").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b =
      g.tensor(TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  // Fully hardcoded MLIR — no placeholders at all.  Only {FUNC_NAME} is used
  // so that the function name matches the emitter's func.call target.
  std::string addMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[4],f32>,
                                   %arg1: !torch.vtensor<[4],f32>)
                                   -> !torch.vtensor<[4],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int
        -> !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
  }
)";

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(addMlir).setNumOutputs(1);

  auto outs = g.customOp({a, b}, addAttr);
  outs[0]
      ->setDim({4})
      .setStride({1})
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
