// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies custom op ASM emission with setIsStatic(true):
//   - Module-scope function uses static shapes (not dynamic ?)
//   - Casts are static -> static (identity no-ops)
//   - func.call uses static types in signature

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
// CHECK:         func.func @main(
// CHECK-SAME:      %my_add_OUT_0_: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f32>
// CHECK-SAME:      %b: !torch.vtensor<[4],f32>
// CHECK:           %a_my_add_i0_perm = torch.aten.permute %a
// CHECK:           %a_my_add_i0_dyn = torch.tensor_static_info_cast %a_my_add_i0_perm : !torch.vtensor<[4],f32> to !torch.vtensor<[4],f32>
// CHECK:           %b_my_add_i1_perm = torch.aten.permute %b
// CHECK:           %b_my_add_i1_dyn = torch.tensor_static_info_cast %b_my_add_i1_perm : !torch.vtensor<[4],f32> to !torch.vtensor<[4],f32>
// CHECK:           %my_add_OUT_0_my_add_dyn = func.call @my_add(%a_my_add_i0_dyn, %b_my_add_i1_dyn) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
// CHECK:           %my_add_OUT_0_my_add_perm = torch.tensor_static_info_cast %my_add_OUT_0_my_add_dyn : !torch.vtensor<[4],f32> to !torch.vtensor<[4],f32>
// CHECK:           %my_add_OUT_0 = torch.aten.permute %my_add_OUT_0_my_add_perm
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

  // Static MLIR: shapes are baked in as [4] instead of [?] placeholders.
  std::string addMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[4],{IN0_DTYPE}>,
                                   %arg1: !torch.vtensor<[4],{IN1_DTYPE}>)
                                   -> !torch.vtensor<[4],{OUT0_DTYPE}> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[4],{IN0_DTYPE}>, !torch.vtensor<[4],{IN1_DTYPE}>, !torch.int
        -> !torch.vtensor<[4],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[4],{OUT0_DTYPE}>
  }
)";

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(addMlir).setNumOutputs(1).setIsStatic(true);

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
