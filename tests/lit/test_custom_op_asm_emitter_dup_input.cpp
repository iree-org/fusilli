// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies duplicate-input custom op ASM emission:
//   - Same tensor passed to both input slots
//   - Per-input indexed suffixes (_i0, _i1) produce unique SSA names

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @my_add
// CHECK:         func.func @main(
// CHECK-SAME:      %{{[^:]*}}: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f32>
// CHECK:           %a_my_add_i0_perm = torch.aten.permute %a
// CHECK:           %a_my_add_i0_dyn = torch.tensor_static_info_cast %a_my_add_i0_perm : !torch.vtensor<[4],f32> to !torch.vtensor<[?],f32>
// CHECK:           %a_my_add_i1_perm = torch.aten.permute %a
// CHECK:           %a_my_add_i1_dyn = torch.tensor_static_info_cast %a_my_add_i1_perm : !torch.vtensor<[4],f32> to !torch.vtensor<[?],f32>
// CHECK:           %my_add_OUT_0_my_add_dyn = func.call @my_add(%a_my_add_i0_dyn, %a_my_add_i1_dyn)
// CHECK:           %my_add_OUT_0_my_add_perm = torch.tensor_static_info_cast %my_add_OUT_0_my_add_dyn : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
// CHECK:           torch.overwrite.tensor.contents %{{.*}} overwrites %{{.*}}
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include <iostream>
#include <string>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("custom_op_asm_emitter_dup_input").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  std::string addMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>,
                                   %arg1: !torch.vtensor<[?],{IN1_DTYPE}>)
                                   -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[?],{IN0_DTYPE}>, !torch.vtensor<[?],{IN1_DTYPE}>, !torch.int
        -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(addMlir).setNumOutputs(1);

  auto outs = g.customOp({a, a}, addAttr);
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

  std::cout << *asmOrErr << std::endl;
  return 0;
}
